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
#include "Flash.hpp"   

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

    /* Read person registration data from c++ header file */
    int32_t ret;
    ret = read_collection_from_file(faceEmbeddingCollection);
    faceEmbeddingCollection.PrintEmbeddings();

    // Add embeddings for a persons (Inference)
    // faceEmbeddingCollection.AddAvgEmbedding("Dinusha", {0.709523, 0.421853, -0.548193, 1.16521, -1.38366, 1.25985, -0.573061, 0.947167, -0.508239, -0.0410354, 0.947351, -1.33453, -0.0174005, -0.881312, 0.686299, -0.0964015, -0.695803, -1.23779, 0.851766, -0.324521, 0.928323, 0.759157, 0.222395, 0.210704, -1.31694, 0.072176, 0.616357, -0.866573, 0.853211, -1.04472, -0.528394, -1.40772, 0.601606, -1.3379, 0.626486, -1.13998, 0.594444, 0.91627, -0.864641, 0.337049, -0.0959521, 0.899826, -1.34723, 0.665579, 0.233292, -0.347827, 1.04125, -0.163949, 0.107878, -0.697345, -0.445537, 0.507816, 0.346021, -0.848526, -0.859321, 0.902601, 0.472195, 0.878496, -0.66742, -0.535055, 1.19782, 1.17391, 0.0749917, -0.0718886});
    // faceEmbeddingCollection.AddAvgEmbedding("Ruchini", {1.24106, 0.309315, -0.442158, 1.10367, -0.0382379, 0.420448, -1.26989, 1.18814, -1.13648, -0.370435, 1.2145, -1.38531, 0.696297, -1.37513, -0.550385, 0.310117, -0.00409993, -0.848525, 1.11894, 0.15799, 1.26183, 0.0503241, -0.106012, 0.380558, -1.03094, -0.711252, 0.164585, -0.290152, 0.843754, -0.878273, -0.731137, -1.30663, -0.638647, 0.479203, 0.0775862, -0.709969, 0.668754, 0.264593, 0.353155, 1.02679, -1.08797, -0.148208, -1.37452, 0.967676, 0.0775862, 0.581957, -0.111905, -0.132472, 0.753408, -0.977339, 0.82888, 0.73685, 1.04656, 0.411709, -0.813305, -0.979624, -1.23204, -0.366105, -0.380819, -0.0628941, 1.05405, 1.12614, 1.08779, -0.513341});

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