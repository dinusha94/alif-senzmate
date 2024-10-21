/* This file was ported to work on Alif Semiconductor Ensemble family of devices. */

/* Copyright (C) 2023 Alif Semiconductor - All Rights Reserved.
 * Use, distribution and modification of this code is permitted under the
 * terms stated in the Alif Semiconductor Software License Agreement
 *
 * You should have received a copy of the Alif Semiconductor Software
 * License Agreement with this file. If not, please write to:
 * contact@alifsemi.com, or visit: https://alifsemi.com/license
 *
 */

/*
 * Copyright (c) 2022 Arm Limited. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "hal.h"                      /* Brings in platform definitions. */
#include "InputFiles.hpp"             /* For input images. */
#include "YoloFastestModel.hpp"       /* Model class for running inference. */
#include "UseCaseHandler.hpp"         /* Handlers for different user options. */
#include "UseCaseCommonUtils.hpp"     /* Utils functions. */
#include "log_macros.h"             /* Logging functions */
#include "BufAttributes.hpp"        /* Buffer attributes to be applied */

#include "MobileNetModel.hpp"       /* Model class for running inference. */
#include "FaceEmbedding.hpp" 



namespace arm {
namespace app {
    
    namespace object_detection {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } /* namespace object_detection */

    namespace img_class{
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } // namespace object_recognition
    static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
} /* namespace app */
} /* namespace arm */

/* callback function to handle name strings received from speech recognition process*/
void user_message_callback(char *message)
{
    info("Message received in user callback: %s\n", message);
}

void main_loop()
{
    init_trigger_rx();
    // init_trigger_tx_custom(user_message_callback);


    arm::app::YoloFastestModel det_model;  /* Model wrapper object. */
    arm::app::MobileNetModel recog_model;
    
    /* No need to initiate Classification since we use single camera*/
    if (!alif::app::ObjectDetectionInit()) {
        printf_err("Failed to initialise use case handler\n");
    }

    /* Load the detection model. */
    if (!det_model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::object_detection::GetModelPointer(),
                    arm::app::object_detection::GetModelLen())) {
        printf_err("Failed to initialise model\n");
        return;
    }


    /* Load the recognition model. */
    if (!recog_model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::img_class::GetModelPointer(),
                    arm::app::img_class::GetModelLen(),
                    det_model.GetAllocator())) {
        printf_err("Failed to initialise recognition model\n");
        return;
    }


    /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;

    arm::app::Profiler profiler{"object_detection"};
    // arm::app::Profiler profiler{"img_class"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::Model&>("det_model", det_model);
    caseContext.Set<arm::app::Model&>("recog_model", recog_model);
    
    // Save the name of the person (later this will be the  name given from the asr process by user_message_callback())
    // std::string myName = "Dinusha";
    // caseContext.Set<std::string&>("my_name", myName);
 
    // // Dynamically allocate the vector on the heap
    // auto croppedImages = std::make_shared<std::vector<std::vector<uint8_t>>>();
    // // Set the context to use the heap-allocated object
    // caseContext.Set<std::shared_ptr<std::vector<std::vector<uint8_t>>>>("cropped_images", croppedImages);

    // Dynamically allocate the vector on the heap to hold CroppedImageData
    // auto croppedImages = std::make_shared<std::vector<CroppedImageData>>();
    // Set the context to use the heap-allocated object
    // caseContext.Set<std::shared_ptr<std::vector<CroppedImageData>>>("cropped_images", croppedImages);

    // Set the context to save the facial embeddings and corresponding name
    // FaceEmbeddingCollection faceEmbeddingCollection;
    // caseContext.Set<FaceEmbeddingCollection&>("face_embedding_collection", faceEmbeddingCollection);


    /* Loop. */
    do {
        alif::app::ObjectDetectionHandler(caseContext);
        // alif::app::ClassifyImageHandler(caseContext);
    } while (1);
}
