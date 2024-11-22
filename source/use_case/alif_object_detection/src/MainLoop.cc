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
#include "delay.h"

#include "FaceEmbedding.hpp"        /* Class for face embedding related functions */
#include "Flash.hpp"                /* Class for external flash memory operations */
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


void main_loop()
{   
    /* Trigger when a name received from asr */
    init_trigger_tx_custom(user_message_callback);

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
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::Model&>("det_model", det_model);
    caseContext.Set<arm::app::Model&>("recog_model", recog_model);
     
    // Dynamically allocate the vector on the heap to hold CroppedImageData
    auto croppedImages = std::make_shared<std::vector<alif::app::CroppedImageData>>();
    caseContext.Set<std::shared_ptr<std::vector<alif::app::CroppedImageData>>>("cropped_images", croppedImages);

    // Set the context to save the facial embeddings and corresponding name
    FaceEmbeddingCollection faceEmbeddingCollection;
    caseContext.Set<FaceEmbeddingCollection&>("face_embedding_collection", faceEmbeddingCollection);

    // Collection to load stored face embeddings
    FaceEmbeddingCollection stored_collection;

    // flag to notify face detection
    bool faceFlag = false;
    caseContext.Set<bool>("face_detected_flag", faceFlag);

    // flag to notify button press event
    caseContext.Set<bool>("buttonflag", false);

    // Hardcoded name
    std::string myName = "";
    caseContext.Set<std::string&>("my_name", myName);

    bool avgEmbFlag = false;
    int loop_idx = 0;
    int32_t ret;

    while(1) {

        alif::app::ObjectDetectionHandler(caseContext);

        // speech recognition method
        if (receivedMessage[0] != '\0') {
            info("Name received: %s\n", receivedMessage);
            myName = receivedMessage;
            caseContext.Set<std::string&>("my_name", myName);
            memset(receivedMessage, '\0', MAX_MESSAGE_LENGTH); // clear the massage buffer
        }

        /* extract the facial embedding and register the person */
        if (caseContext.Get<bool>("face_detected_flag") && !myName.empty()) { 
            avgEmbFlag = true;
            info("registration .. \n");

            if (avgEmbFlag && (loop_idx < 5)){
                info("Averaging embeddings .. \n");
                alif::app::ClassifyImageHandler(caseContext); 
                sleep_or_wait_msec(1000); /* wait for possible pose changes */
                loop_idx ++; 
            }else {
                avgEmbFlag = false;
                loop_idx = 0;

                // average the embedding fro the myName
                faceEmbeddingCollection.CalculateAverageEmbeddingAndSave(myName);
                info("Averaging finished and saved .. \n");

                faceEmbeddingCollection.PrintEmbeddings();

                /* save embedding data to external flash  */
                ret = flash_send(faceEmbeddingCollection);
                ret = ospi_flash_read_collection(stored_collection);
                stored_collection.PrintEmbeddings();

                caseContext.Set<bool>("face_detected_flag", false); // Reset flag 
                myName.clear();
                caseContext.Set<std::string&>("my_name", myName);
            }
        }      
        
    };
    
}