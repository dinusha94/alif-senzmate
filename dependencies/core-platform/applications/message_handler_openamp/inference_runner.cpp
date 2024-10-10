/*
 * SPDX-FileCopyrightText: Copyright 2022-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use _this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*****************************************************************************
 * Includes
 *****************************************************************************/

#include "inference_runner.hpp"

#include <cstdlib>

#include <ethosu_log.h>

#if defined(ETHOSU)
#include <ethosu_driver.h>
#include <pmu_ethosu.h>
#endif

/*****************************************************************************
 * InferenceRunner
 *****************************************************************************/

InferenceRunner::InferenceRunner(uint8_t *tensorArena,
                                 size_t arenaSize,
                                 MessageHandler::InferenceQueue &_inferenceQueue,
                                 MessageHandler::ResponseQueue &_responseQueue) :
    inferenceQueue(_inferenceQueue),
    responseQueue(_responseQueue), inference(tensorArena, arenaSize) {
    BaseType_t ret = xTaskCreate(inferenceTask, "inferenceTask", 8 * 1024, this, 4, &taskHandle);
    if (ret != pdPASS) {
        LOG_ERR("Failed to create inference task");
        abort();
    }
}

InferenceRunner::~InferenceRunner() {
    vTaskDelete(taskHandle);
}

void InferenceRunner::inferenceTask(void *param) {
    auto _this = static_cast<InferenceRunner *>(param);

    LOG_DEBUG("Starting inference task");

    while (true) {
        Message *message;
        auto ret = _this->inferenceQueue.receive(message);
        if (ret) {
            abort();
        }

        auto &rpmsg = message->rpmsg;

        switch (rpmsg.header.type) {
        case EthosU::ETHOSU_CORE_MSG_INFERENCE_REQ: {
            _this->handleInferenceRequest(message->src, rpmsg.header.msg_id, rpmsg.inf_req);
            break;
        }
        default: {
            LOG_WARN("Unsupported message for inference runner. type=%lu", rpmsg.header.type);
        }
        }

        delete message;
    }
}

void InferenceRunner::handleInferenceRequest(const uint32_t src,
                                             const uint64_t msgId,
                                             const EthosU::ethosu_core_msg_inference_req &request) {
    auto message =
        new Message(src, EthosU::ETHOSU_CORE_MSG_INFERENCE_RSP, msgId, sizeof(EthosU::ethosu_core_msg_inference_rsp));
    auto &response = message->rpmsg.inf_rsp;

    // Setup PMU configuration
    response.pmu_cycle_counter_enable = request.pmu_cycle_counter_enable;

    for (int i = 0; i < ETHOSU_CORE_PMU_MAX; i++) {
        response.pmu_event_config[i] = request.pmu_event_config[i];
    }

    // Run inference
    auto job    = makeInferenceJob(request, response);
    auto failed = inference.runJob(job);

    // Send response rpmsg
    response.ofm_count = job.output.size();
    response.status    = failed ? EthosU::ETHOSU_CORE_STATUS_ERROR : EthosU::ETHOSU_CORE_STATUS_OK;

    for (size_t i = 0; i < job.output.size(); ++i) {
        response.ofm_size[i] = job.output[i].size;
    }

    responseQueue.send(message);
}

InferenceProcess::InferenceJob InferenceRunner::makeInferenceJob(const EthosU::ethosu_core_msg_inference_req &request,
                                                                 EthosU::ethosu_core_msg_inference_rsp &response) {
    InferenceProcess::InferenceJob job;

    job.networkModel =
        InferenceProcess::DataPtr(reinterpret_cast<void *>(request.network.buffer.ptr), request.network.buffer.size);

    for (uint32_t i = 0; i < request.ifm_count; ++i) {
        job.input.push_back(
            InferenceProcess::DataPtr(reinterpret_cast<void *>(request.ifm[i].ptr), request.ifm[i].size));
    }

    for (uint32_t i = 0; i < request.ofm_count; ++i) {
        job.output.push_back(
            InferenceProcess::DataPtr(reinterpret_cast<void *>(request.ofm[i].ptr), request.ofm[i].size));
    }

    job.externalContext = &response;

    return job;
}

#if defined(ETHOSU)
extern "C" {

void ethosu_inference_begin(ethosu_driver *drv, void *userArg) {
    LOG_DEBUG("");

    auto response = static_cast<EthosU::ethosu_core_msg_inference_rsp *>(userArg);

    // Calculate maximum number of events
    const int numEvents = std::min(static_cast<int>(ETHOSU_PMU_Get_NumEventCounters()), ETHOSU_CORE_PMU_MAX);

    // Enable PMU
    ETHOSU_PMU_Enable(drv);

    // Configure and enable events
    for (int i = 0; i < numEvents; i++) {
        ETHOSU_PMU_Set_EVTYPER(drv, i, static_cast<ethosu_pmu_event_type>(response->pmu_event_config[i]));
        ETHOSU_PMU_CNTR_Enable(drv, 1u << i);
    }

    // Enable cycle counter
    if (response->pmu_cycle_counter_enable) {
        ETHOSU_PMU_PMCCNTR_CFG_Set_Stop_Event(drv, ETHOSU_PMU_NPU_IDLE);
        ETHOSU_PMU_PMCCNTR_CFG_Set_Start_Event(drv, ETHOSU_PMU_NPU_ACTIVE);

        ETHOSU_PMU_CNTR_Enable(drv, ETHOSU_PMU_CCNT_Msk);
        ETHOSU_PMU_CYCCNT_Reset(drv);
    }

    // Reset all counters
    ETHOSU_PMU_EVCNTR_ALL_Reset(drv);
}

void ethosu_inference_end(ethosu_driver *drv, void *userArg) {
    auto response = static_cast<EthosU::ethosu_core_msg_inference_rsp *>(userArg);

    // Get cycle counter
    if (response->pmu_cycle_counter_enable) {
        response->pmu_cycle_counter_count = ETHOSU_PMU_Get_CCNTR(drv);
    }

    // Calculate maximum number of events
    const int numEvents = std::min(static_cast<int>(ETHOSU_PMU_Get_NumEventCounters()), ETHOSU_CORE_PMU_MAX);

    // Get event counters
    int i;
    for (i = 0; i < numEvents; i++) {
        response->pmu_event_count[i] = ETHOSU_PMU_Get_EVCNTR(drv, i);
    }

    for (; i < ETHOSU_CORE_PMU_MAX; i++) {
        response->pmu_event_config[i] = 0;
        response->pmu_event_count[i]  = 0;
    }

    // Disable PMU
    ETHOSU_PMU_Disable(drv);
}
}

#endif
