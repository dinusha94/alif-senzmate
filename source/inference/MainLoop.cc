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

    // Add embeddings for a persons (Inference)
    // faceEmbeddingCollection.AddEmbedding("Dinusha", {108, 54, 25, 92, -124, 123, -126, 127, 96, -61, 8, -124, 16, -25, 114, 47, 
    //                                                     112, -116, -25, 8, 114, 116, -67, -8, -116, -25, 54, 16, 25, -105, -116, -125, 
    //                                                     126, -121, 114, -99, -67, 127, -123, 102, 124, 118, -128, -61, 126, 40, 121, 
    //                                                     102, -105, -125, 54, -8, 0, -127, -117, 78, 96, -16, 126, -32, 127, 116, -112, 
    //                                                     120});

    // faceEmbeddingCollection.AddEmbedding("Dinusha", {16, 121, -83, 123, -123, 127, -32, 105, -83, 73, 126, -102, 
    //                                                     -83, -125, 61, 25, -114, -83, 127, 0, 125, 88, 105, 32, -114, 
    //                                                     16, 96, -110, 102, -78, -16, -127, 8, -124, 73, -108, 118, 
    //                                                     78, 32, -73, -99, 54, -126, 114, 92, -32, 92, -116, 40, -54, 
    //                                                     -99, 120, 73, -114, -47, 99, 32, 120, -120, 0, 105, 127, 99, -67});

    faceEmbeddingCollection.AddEmbedding("Dinusha", {47, -121, -99, 127, -127, 122, -61, 67, -92, -61, 127, -127, 114, -32, 32, -125, 
        -108, -119, 110, -83, 32, 25, 32, 114, -123, -61, 125, -112, 125, -105, 0, -128, 
        54, -108, 108, -116, 83, 73, -110, 96, 54, 96, -128, 92, -32, 8, 78, 83, 99, 
        -127, -32, 121, -47, -108, -40, 78, -32, 127, -108, -47, 127, 92, -47, 16});
                                                        
    faceEmbeddingCollection.AddEmbedding("Alice", {55, 108, 83, -8, -105, 25, 16, 32, 61, 99, 55, -83, -55, -127, 124, 47,
                                                        -117, 78, 61, -105, -25, 25, 105, -102, -125, -25, -73, -119, -73, 55,
                                                        -88, -108, 25, -47, -73, 78, 112, -25, 118, -54, -124, 108, -16, 25, -40,
                                                        55, 25, -40, 32, 25, 5, 78, 32, 40, 102, -121, 55, -126, -67, 8, -78,
                                                        25, 0, -99});

    // faceEmbeddingCollection.AddEmbedding("Ruchini", {126, 102, -40, 102, 105, 123, -114, 116, -78, 54, 120, -126, 
    //                                                     78, -128, -32, 110, 83, -54, 99, -61, 126, 108, -105, -16, 
    //                                                     -61, -83, -47, 78, 47, -40, -126, -118, -32, -16, 121, -122, 
    //                                                     114, 122, 125, 112, -112, -8, -122, 92, 121, 25, -112, -110, 
    //                                                     78, -96, 54, 120, 102, -16, -117, -67, -99, -110, 54, 118, 
    //                                                     125, 119, 99, -25});

    faceEmbeddingCollection.AddEmbedding("Ruchini", {123, -32, -32, 121, -102, -32, -116, 123, -127, -114, 124, -126, 67, -122, -61, -40, 
        -73, -96, 127, 105, 127, -88, 99, 102, -124, -40, 92, -123, 127, -116, 0, -119, -78, 
        121, -96, 0, 25, -61, -47, 96, -83, -8, -128, 105, -96, 99, 105, 99, 78, -78, 117, 
        32, 110, 108, -25, -108, -124, 54, -116, -120, 88, 108, 121, -61});

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
