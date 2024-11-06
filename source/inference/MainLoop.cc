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

void delay_ms(int milliseconds) {
    // Simple busy wait; not accurate, just for demonstration
    for (volatile int i = 0; i < milliseconds * 1000; ++i) {
        // Empty loop for delay
    }
}

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

/*
const int MAX_MESSAGE_LENGTH = 256;
char receivedMessage[MAX_MESSAGE_LENGTH];

void user_message_callback(char *message) {
    // Store the received message in the global variable
    strncpy(receivedMessage, message, MAX_MESSAGE_LENGTH - 1);
    receivedMessage[MAX_MESSAGE_LENGTH - 1] = '\0'; // Ensure null-termination
    info("Message received in user callback..............................: %s\n", receivedMessage);
}
*/

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

    arm::app::Profiler profiler_det{"object_detection"};
    arm::app::Profiler profiler_class{"img_class"};
    caseContext.Set<arm::app::Profiler&>("profiler_det", profiler_det);
    caseContext.Set<arm::app::Profiler&>("profiler_class", profiler_class);
    caseContext.Set<arm::app::Model&>("det_model", det_model);
    caseContext.Set<arm::app::Model&>("recog_model", recog_model);
     
    // Dynamically allocate the vector on the heap to hold CroppedImageData
    auto croppedImages = std::make_shared<std::vector<alif::app::CroppedImageData>>();
    caseContext.Set<std::shared_ptr<std::vector<alif::app::CroppedImageData>>>("cropped_images", croppedImages);

    // Set the context to save the facial embeddings and corresponding name
    FaceEmbeddingCollection faceEmbeddingCollection;

    // Add embeddings for a persons (Inference)
    faceEmbeddingCollection.AddEmbedding("Dinusha", {87, 34, 0, -34, -102, 34, -34, 87, -87, 34, 34, 64, 34, 34, 64, 0, -102, 0, -64, 102, 
                                                        64, -102, 0, 119, 102, 87, 0, 34, 0, 34, 123, 34, -119, 64, -34, 0, 87, -34, -126, -64, 
                                                        87, -34, 34, 34, 64, 34, 64, -64, 0, -34, 34, -87, 113, -113, -102, 113, 0, 34, -113, 
                                                        64, -34, 34, -34, -87, 34, -64, 87, -113, 34, -34, 34, -34, -34, -34, 113, 64, -64, -87, 
                                                        34, 102, -125, 0, 87, 0, 87, 0, -113, 102, 0, 123, -34, 113, 0, 0, 34, -102, 87, -87, 
                                                        -119, -34, 64, -34, -34, -34, 0, 119, -34, 64, 64, -34, 0, 87, 34, 34, 87, 34, -64, 34, 
                                                        113, -34, -102, -64, -64, -102, 102, 34, -34, -87, 34, -34, -64, -87, 87, 113, 113, 0, 
                                                        87, 64, -34, 34, 87, 64, -113, -64, 87, 64, -102, 119, -113, -34, -87, 87, -34, -87, 34, 
                                                        -34, -87, -87, -113, 119, -87, -34, -34, -113, -113, 102, -34, 0, 34, 34, 64, -102, -64, 
                                                        123, 102, -102, 87, 64, 64, 64, 0, -113, 0, -64, 0, 0, 0, 87, 119, 87, 102, 87, -119, 
                                                        0, 0, 64, -102, 87, 34, 34, -34, 87, 87, -113, 0, 64, 34, 64, -34, 125, -34, -123, 87, 
                                                        -64, -34, -34, 87, 34, 87, -34, 87, -34, -102, 87, 123, 119, 34, -102, -64, 34, 0, -64, 
                                                        -102, 64, 64, 123, -34, -87, 123, -64, -87, 64, -34, 0, -34, 34, 0, -119, 102, 87, 87, 
                                                        0, -102, -34, 34, 123, -34, -64, 64, 34, 0, 34, 102, -102, -119, -102, -87, 102, -102, 
                                                        113, -102, -34, 64, 0, 34, -123, 87, -64, 87, -102, 102, -119, 0, -64, -64, -125, -119, 
                                                        113, -34, 0, 64, -64, -102, -64, -123, -64, -64, -87, 0, -34, 102, -87, -64, -102, 64, 
                                                        -34, 113, 64, 34, 87, 64, 34, -126, 34, 113, -123, -87, 0, 119, 0, 64, -87, 0, 34, 34, 
                                                        34, 34, -64, -64, 102, 34, 102, 0, 123, -64, -87, -64, -87, 127, 123, -34, 64, 0, 119, 
                                                        102, -64, -113, 0, 34, -64, 0, 64, -113, 126, -113, 119, -119, -64, 102, 102, 102, 0, 
                                                        34, -64, 0, 102, 64, -64, -113, 64, -34, 0, 34, 0, 34, 34, 113, -34, -64, 34, 0, -34, 
                                                        -64, 102, -34, -64, 64, -34, -102, -102, -125, 0, 34, 0, -34, -34, 64, -102, 102, -34, 
                                                        0, -64, -34, -102, 64, 87, 123, 0, -64, 64, 0, 87, 0, -102, 0, -34, -34, -113, 102, 
                                                        -102, -102, -34, -125, -87, 64, 64, 64, 87, -34, 64, -87, 123, 87, -34, -113, 0, -113, 
                                                        -102, 119, -64, -87, 34, -64, 34, -34, -102, 123, 113, -64, -64, 0, -113, 0, 34, 64, 
                                                        -113, -34, -64, 113, -34, 102, 87, 87, -34, -64, -64, -113, 64, 64, 0, 102, 0, 34, -64, 
                                                        119, 0, -64, 0, 34, 64, 87, -123, -64, 34, -87, -34, -34, -87, 64, -34, 64, 0, -87, 0, 
                                                        87, 64, -119, 113, -119, 126, -113, 87, 102, 0, -123, 34, -34, 0, 34, 123, 87, -102});

    faceEmbeddingCollection.AddEmbedding("Ruchini", {87, 64, -34, 0, -34, 34, 64, 0, -34, -64,
                                                            34, 102, -34, -34, 0, 0, -64, -64, -64, 34,
                                                            34, 87, -87, 64, 34, 0, 64, 64, -87, 64,
                                                            34, -34, -102, -34, -87, 34, -34, -34, 87, -34,
                                                            -34, -34, 64, 34, 34, 0, 34, -64, -34, 34,
                                                            34, 34, 64, -64, -64, 87, 34, -34, 34, 34,
                                                            34, -34, -34, -34, 0, 34, -64, -64, -34, -102,
                                                            64, 0, -64, 34, 34, 0, 0, -34, 34, 34,
                                                            34, -64, -87, 0, 64, 64, -64, 0, 64, -34,
                                                            -87, -64, 64, 0, 34, -64, 0, -34, -64, 34,
                                                            34, 0, 64, 64, 0, 87, 34, 34, 34, -64,
                                                            34, 0, 64, -34, 34, 64, 0, 0, 34, 0,
                                                            -64, 64, -34, -34, 0, 34, 0, -34, 0, -34,
                                                            34, -87, 64, 34, 102, 0, 0, 34, 64, -34,
                                                            34, 64, -34, -34, -34, 0, -64, -34, -34, -34,
                                                            64, -87, 0, 0, 34, 34, 0, -87, -64, 64,
                                                            34, 34, -34, 0, 0, -34, 64, -64, 34, -34,
                                                            -34, -87, 0, 87, -87, -64, 102, 64, 34, 0,
                                                            34, -64, -34, 87, -34, 0, 0, 0, 0, 34,
                                                            87, 0, -34, 34, 0, -34, 34, 0, 0, -87,
                                                            0, 34, 64, -119, 64, 0, 64, -64, -87, -34,
                                                            -64, -64, 64, -113, -34, 64, 113, 87, 102, -64,
                                                            34, 87, 0, 34, 34, 87, -34, -34, 0, 0,
                                                            -64, -87, 34, -34, 0, 87, 0, 0, 34, -64,
                                                            64, 0, 0, -34, -34, -87, 0, 34, 0, 64,
                                                            34, 0, -64, -34, 87, 0, 34, -64, 64, 34,
                                                            34, 64, 64, 64, -34, -64, 0, -87, -123, 0,
                                                            34, -64, 87, -34, 64, -34, 87, -64, -64, -34,
                                                            0, 0, -34, -34, 0, -87, -64, 0, 34, -34,
                                                            34, 0, 0, -34, 0, -34, 0, 34, 0, 34,
                                                            34, -34, 0, -34, 87, 34, 34, 0, 34, 64,
                                                            34, -34, -64, 34, 87, -34, -87, 64, 87, 34,
                                                            0, 0, -34, 64, 34, -64, 34, -64, -34, 34,
                                                            -87, 64, -64, -34, -34, 64, 0, -34, 34, -64,
                                                            -87, 64, 64, 0, 34, 34, 0, -64, 0, -64,
                                                            -34, 0, -34, 0, -64, 34, 0, -34, 64, -87,
                                                            0, 34, 34, -34, 0, 0, -64, -34, -102, -34,
                                                            -34, 34, 34, 34, 34, -34, 113, 0, 34, -34,
                                                            64, 0, -64, 64, 34, -34, 64, -64, 34, 0,
                                                            -87, 34, 34, 0, -64, -34, 64, 34, 34, 0,
                                                            0, 34, 0, -34, 0, 34, -113, 0, 0, 87,
                                                            -34, 34, 0, 0, 0, 123, -102, 0, 87, -34,
                                                            64, 0, -87, -64, -34, 0, -102, 0, 0, 64,
                                                            0, 34, 0, 34, 34, -34, -34, -64, 34, 0,
                                                            -34, 87, 0, 34, 0, -102, 0, -34, 34, -34,
                                                            -87, 0, -34, -34, 64, -34, -34, -34, 87, 0,
                                                            0, 0, 102, 64, -87, 0, 34, 87, 0, 0,
                                                            64, 0, 64, -34, -34, -34, 0, -34, 34, 0,
                                                            34, -87, -64, -34, 64, 0, -64, 0, 0, -34,
                                                            0, 64, 0, -87, 64, 64, -64, 0, -64, 113,
                                                            0, -34, 87, 64, 0, 0, -64, 87, 64, 34,
                                                            -34, -64});
    
    faceEmbeddingCollection.AddEmbedding("Alice", {2, 2, 5, -87, -64, 102, -3, 125, 87, 0, 102, 64, 102, 113, -87, -34,
                                                    -113, 64, -87, 87, 64, -64, -34, 0, 64, 113, -34, 87, 64, 87, 113, 34,
                                                    -87, 64, -64, 34, 64, 34, -123, -87, 123, 0, 64, 34, 34, 64, 123, -87,
                                                    34, -125, -87, -64, 64, -55, -113, 102, -87, 0, -64, 0, -119, 34, 34,
                                                    -64, -34, 0, 34, -102, -123, -34, -64, 0, -87, -34, 113, 34, -64, -64,
                                                    119, 0, -128, 102, 64, 34, 119, -102, -113, 64, 113, 123, 87, 127, 87,
                                                    87, -64, 0, 102, -87, -113, 0, 0, -87, -64, 0, 64, 123, -64, 119, 64,
                                                    0, -34, 55, -55, -64, 119, -64, -34, 64, 102, -34, -102, -113, 0, -64,
                                                    55, 34, -34, -34, -87, 34, -64, -34, 87, 87, 102, -34, 123, 102, 34, 0,
                                                    113, 64, -64, -64, 113, 64, -126, 126, -5, 5, -5, 64, 34, -123, 87,
                                                    0, -87, -5, -5, 5, -5, 5, -5, -113, -64, 102, 0, -34, 0, -87,
                                                    87, -87, 34, 102, 123, -102, 87, 34, 34, 0, 0, -119, 34, 0, -34, 64, 34,
                                                    64, 5, 64, 102, 113, -55, 0, -64, 5, -87, 102, 34, 102, 34, 34, 87,
                                                    -102, -113, 0, 119, 64, 55, 127, -64, -119, 119, -87, 0, 0, 0, -64, 113,
                                                    -64, 55, 113, -34, 0, 119, 119, 66, -64, -5, -87, -64, -87, -119, 55,
                                                    87, 87, 87, -64, 127, -55, -126, 64, -55, -34, 0, 34, 0, -127, 34, 64,
                                                    102, -102, -5, -87, 102, 119, 34, 55, -34, 87, 64, 102, 64, -102, -102,
                                                    -113, -113, 64, -34, 87, -119, -34, 102, -34, 0, -127, 113, -64, 87, -102,
                                                    64, -43, 119, -64, 34, -125, -113, 113, -34, 0, 34, -102, -113, -34, -102,
                                                    -34, -34, -87, 102, -113, 55, -102, 0, -64, -34, -87, 87, 64, 0, 64, -34,
                                                    34, -55, -55, 55, -55, -55, 55, 55, 55, 55, -34, 0, 34, -64, 64, 34,
                                                    -87, -55, -34, 34, 87, 64, 123, -64, -87, -34, -64, 127, 126, 34, 102, 0,
                                                    119, 64, -34, -55, 55, 126, 87, -55, -55, -55, 55, -55, 125, -119, -34,
                                                    64, 127, 127, 34, 0, 34, 55, 55, 0, -87, -64, 87, 87, 0, 123, 64, 64, 64,
                                                    119, -64, -119, 34, 64, 34, -87, 6, -119, -34, -34, 64, -34, -55, -102,
                                                    -34, 64, 0, 34, -64, 87, -126, 64, 64, -64, -87, 64, -113, 87, 64, 119, 87,
                                                    34, 34, 102, 87, -34, -119, 6, 0, 64, -102, 102, -34, -87, -64, -113, -34,
                                                    0, 34, 119, 102, -113, 0, -34, 125, 66, 34, -55, 64, -102, -113, 119, -87,
                                                    -34, 64, -87, 34, 0, -6, 102, 119, -66, -64, 87, -87, 64, -34, -64, -64,
                                                    -64, 34, 123, 64, 67, 55, 64, 87, -66, -87, -113, 0, 126, 34, 87, 64,
                                                    123, -64, 102, 0, 0, 87, -34, 87, 66, -123, 34, 64, -64, 0, -64, -64, 0,
                                                    -87, -34, 34, -87, 0, 125, 113, -66, 55, -87, 102, -113, 64, 87, -102, -102,
                                                    -87, 34, 0, 55, 5, 55, -55});

    caseContext.Set<FaceEmbeddingCollection&>("face_embedding_collection", faceEmbeddingCollection);

    bool faceFlag = false;
    caseContext.Set<bool>("face_detected_flag", faceFlag);

    // Hardcoded person data to test the registration
    // std::string myName = "Dinusha";
    // caseContext.Set<std::string&>("my_name", myName);
    std::string whoAmI = "";
    caseContext.Set<std::string>("person_id", whoAmI);

    
    /*
    do {
        // Check if there's a new message
        if (strlen(receivedMessage) > 0) {
            // Process the received message
            info("Processing message in main loop.................................: %s\n", receivedMessage);
            std::string myName = receivedMessage; // Create a std::string from the received message
            caseContext.Set<std::string&>("my_name", myName);
            // Clear the received message after processing
            receivedMessage[0] = '\0'; // Reset the message
            break;
        }
    }while (1);
    */
    

    /* Registration Loop. */
    // do {
    //     alif::app::ObjectDetectionHandler(caseContext);

    //     if (caseContext.Get<bool>("face_detected_flag")) {
    //         alif::app::ClassifyImageHandler(caseContext);  // Run feature extraction
    //         caseContext.Set<bool>("face_detected_flag", false); // Reset flag 
    //         // delay_ms(1000);
    //         break; // exit the loop
    //     }
    // } while (1);

    /* Inference Loop. */
     do {
        alif::app::ObjectDetectionHandler(caseContext);
        alif::app::ClassifyImageHandler(caseContext);  // Run feature extraction
    } while (1);
}