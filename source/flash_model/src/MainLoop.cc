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

#include "ExtLoader.hpp"
#include "uart_tracelib.h"

#include "FaceEmbedding.hpp"        /* Face embedding class */
#include "Flash.hpp"                /* External flash store class */

namespace arm {
namespace app {
    static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
    namespace object_detection {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } /* namespace object_detection */
} /* namespace app */
} /* namespace arm */

void convert_uint8_to_uint16(const uint8_t *input_buffer, size_t input_size, uint16_t *output_buffer) {
    size_t output_size = input_size / 2;
    for (size_t i = 0; i < output_size; i++) {
        output_buffer[i] = (input_buffer[2 * i + 1] << 8) | input_buffer[2 * i];
    }
}

void main_loop()
{
    init_trigger_rx();

    arm::app::YoloFastestModel model;  /* Model wrapper object. */

    // if (!alif::app::ObjectDetectionInit()) {
    //     printf_err("Failed to initialise use case handler\n");
    // }

    // /* Load the model. */
    // if (!model.Init(arm::app::tensorArena,
    //                 sizeof(arm::app::tensorArena),
    //                 arm::app::object_detection::GetModelPointer(),
    //                 arm::app::object_detection::GetModelLen())) {
    //     printf_err("Failed to initialise model\n");
    //     return;
    // }

    // /* Instantiate application context. */
    // arm::app::ApplicationContext caseContext;

    // arm::app::Profiler profiler{"object_detection"};
    // caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    // caseContext.Set<arm::app::Model&>("model", model);

    bool start = true;
    int32_t ret;
    
    const uint32_t baseAddress = 0xC0000000;
    const size_t chunkSize = 32;
    size_t totalBytes = 0;

    uint32_t currentAddress = baseAddress; // Start at the base address

    uint8_t receivedArray[chunkSize]; // hold 1KB data if chunkSize is 1024
    size_t input_size = sizeof(receivedArray);
    uint16_t output_buffer[input_size / 2];
    uint32_t buffer_len = input_size / 2; 

    ret = erase_flash();
    info("return erase status: %d \n", ret);

    while(1){

        memset(receivedArray, 0, chunkSize); // set all the elements to 0
            
        for (size_t k = 0; k < chunkSize; ++k) {
            receivedArray[k] =  uart_getchar(); // Receive byte              
        }

        totalBytes += sizeof(receivedArray);

        // Convert the buffer
        convert_uint8_to_uint16(receivedArray, input_size, output_buffer);

        // printf("Write buf:\n");
        // for (size_t i = 0; i < buffer_len; ++i) { 
        //     printf("0x%04x, ",  output_buffer[i]);
        // }
        // printf("\n");

        if (start){
            ret = flash_send_model(baseAddress, output_buffer, buffer_len);
            start = false;  
        }else if (!start){
            ret = flash_send_model(currentAddress, output_buffer, buffer_len);
        }

        if (totalBytes > 13916736){
            info("done \n");
            break;
        }
        
        info("return write status at 0x%8X :%d \n", currentAddress, ret);

        currentAddress = currentAddress + sizeof(output_buffer);
        

        // ret = ospi_flash_read();
        // info("return read status:%d \n", ret);

        info("next_chunk at :  0x%8X \n", currentAddress);

        // break;
    }
    
    // ret = ospi_flash_read();
    // info("return read status:%d \n", ret);

    /* Loop. */
    // do {
    //     alif::app::ObjectDetectionHandler(caseContext);
    // } while (1);
}
