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
#include <iostream>
#include <cstring> 
#include <random>

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

// Global variable to hold the received message
const int MAX_MESSAGE_LENGTH = 256;
char receivedMessage[MAX_MESSAGE_LENGTH];

/* callback function to handle name strings received from speech recognition process*/
void user_message_callback(char *message) {
    strncpy(receivedMessage, message, MAX_MESSAGE_LENGTH - 1);
    receivedMessage[MAX_MESSAGE_LENGTH - 1] = '\0'; // Ensure null-termination
    info("Message received in user callback: %s\n", message);
}



std::string pickRandomName(const std::vector<std::string>& names, std::mt19937& generator) {
    std::uniform_int_distribution<> dist(0, names.size() - 1);
    return names[dist(generator)];
}

std::vector<std::string> nameList = {
        "Alice", "Bob", "Charlie", "David", "Eve",
        "Frank", "Grace", "Hannah", "Ivy", "Jack"
    };


bool last_btn1 = false; 

bool run_requested_(void)
{
    bool ret = false; // Default to no inference
    bool new_btn1;
    BOARD_BUTTON_STATE btn_state1;

    // Get the new button state (active low)
    BOARD_BUTTON1_GetState(&btn_state1);
    new_btn1 = (btn_state1 == BOARD_BUTTON_STATE_LOW); // true if button is pressed

    // Edge detector - run inference on the positive edge of the button pressed signal
    if (new_btn1 && !last_btn1) // Check for transition from not pressed to pressed
    {
        ret = true; // Inference requested
    }

    // Update the last button state
    last_btn1 = new_btn1;

    return ret; // Return whether inference should be run
}

void main_loop()
{
    // init_trigger_rx();
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
     
    // Dynamically allocate the vector on the heap to hold CroppedImageData
    auto croppedImages = std::make_shared<std::vector<alif::app::CroppedImageData>>();
    caseContext.Set<std::shared_ptr<std::vector<alif::app::CroppedImageData>>>("cropped_images", croppedImages);

    // Set the context to save the facial embeddings and corresponding name
    FaceEmbeddingCollection faceEmbeddingCollection;
    caseContext.Set<FaceEmbeddingCollection&>("face_embedding_collection", faceEmbeddingCollection);

    // flag to notify face detection
    bool faceFlag = false;
    caseContext.Set<bool>("face_detected_flag", faceFlag);

    // flag to notify button press event
    caseContext.Set<bool>("buttonflag", false);

    // Hardcoded name
    std::string myName = "Dinusha";
    caseContext.Set<std::string&>("my_name", myName);

    std::random_device rd;
    std::mt19937 generator(rd());

       
    while(1) {

        alif::app::ObjectDetectionHandler(caseContext);

        // if (receivedMessage[0] != '\0') {
        //     info("Name received: %s\n", receivedMessage);
        //     std::string myName(receivedMessage);
        //     caseContext.Set<std::string&>("my_name", myName);
        // }

        if (run_requested_())
        {
            caseContext.Set<bool>("buttonflag", true);
            std::string randomName = pickRandomName(nameList, generator);
            caseContext.Set<std::string&>("my_name", randomName);            

        }

        if (caseContext.Get<bool>("face_detected_flag")) {
            alif::app::ClassifyImageHandler(caseContext);  // Run feature extraction
            caseContext.Set<bool>("face_detected_flag", false); // Reset flag 
            // continue;
        }

        //  uint32_t startWait = Get_SysTick_Cycle_Count32();
        //         uint32_t waitTime = SystemCoreClock / 1000 * 100; 
        //         while ((Get_SysTick_Cycle_Count32() - startWait) < waitTime) {
        //         }       
       
        
    };
    
}