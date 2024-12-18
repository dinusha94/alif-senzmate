/*
 * SPDX-FileCopyrightText: Copyright 2021 Arm Limited and/or its affiliates <open-source-office@arm.com>
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
#include "Labels.hpp"                /* For label strings. */
#include "UseCaseHandler.hpp"        /* Handlers for different user options. */
#include "Wav2LetterModel.hpp"       /* Model class for running inference. */
#include "UseCaseCommonUtils.hpp"    /* Utils functions. */
#include "AsrClassifier.hpp"         /* Classifier. */
#include "InputFiles.hpp"            /* Generated audio clip header. */
#include "log_macros.h"             /* Logging functions */
#include "BufAttributes.hpp"        /* Buffer attributes to be applied */
#include "hal.h"
#include "delay.h"

#include "ospi_flash.h"

// #include "services_lib_api.h"
// #include "services_main.h"

// extern uint32_t m55_comms_handle;
// m55_data_payload_t mhu_data;

namespace arm {
namespace app {
    static uint8_t  tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;
    namespace asr {
        extern uint8_t* GetModelPointer();
        extern size_t GetModelLen();
    } /* namespace asr */
} /* namespace app */
} /* namespace arm */

enum opcodes
{
    MENU_OPT_RUN_INF_NEXT = 1,       /* Run on next vector. */
    MENU_OPT_RUN_INF_CHOSEN,         /* Run on a user provided vector index. */
    MENU_OPT_RUN_INF_ALL,            /* Run inference on all. */
    MENU_OPT_SHOW_MODEL_INFO,        /* Show model info. */
    MENU_OPT_LIST_AUDIO_CLIPS        /* List the current baked audio clips. */
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

static void DisplayMenu()
{
    printf("\n\n");
    printf("User input required\n");
    printf("Enter option number from:\n\n");
    printf("  %u. Classify next audio clip\n", MENU_OPT_RUN_INF_NEXT);
    printf("  %u. Classify audio clip at chosen index\n", MENU_OPT_RUN_INF_CHOSEN);
    printf("  %u. Run classification on all audio clips\n", MENU_OPT_RUN_INF_ALL);
    printf("  %u. Show NN model info\n", MENU_OPT_SHOW_MODEL_INFO);
    printf("  %u. List audio clips\n\n", MENU_OPT_LIST_AUDIO_CLIPS);
    printf("  Choice: ");
    fflush(stdout);
}

// static void send_name(std::string name)
// {
    
//     mhu_data.id = 3; // id for senzmate app

//     info("******************* send_name : %s \n", name.c_str());
//     strcpy(mhu_data.msg, name.c_str());
//     __DMB();
//     SERVICES_send_msg(m55_comms_handle, &mhu_data);
        
// }

/** @brief   Verify input and output tensor are of certain min dimensions. */
static bool VerifyTensorDimensions(const arm::app::Model& model);

/* Buffer to hold kws audio samples */
#define AUDIO_SAMPLES_KWS 32000 // 1 second samples @ 16kHz
#define AUDIO_STRIDE_KWS 16000 
static int16_t audio_inf_kws[AUDIO_SAMPLES_KWS + AUDIO_STRIDE_KWS];

void main_loop()
{
    
    init_trigger_tx();
    
    arm::app::Wav2LetterModel model;  /* Model wrapper object. */

    /* Load the model. */
    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::asr::GetModelPointer(),
                    arm::app::asr::GetModelLen())) {
        printf_err("Failed to initialise model\n");
        return;
    } else if (!VerifyTensorDimensions(model)) {
        printf_err("Model's input or output dimension verification failed\n");
        return;
    }

    // /* Instantiate application context. */
    arm::app::ApplicationContext caseContext;
    std::vector <std::string> labels;
    GetLabelsVector(labels);
    arm::app::AsrClassifier classifier;  /* Classifier wrapper object. */

    arm::app::Profiler profiler{"asr"};
    caseContext.Set<arm::app::Profiler&>("profiler", profiler);
    caseContext.Set<arm::app::Model&>("model", model);
    // caseContext.Set<uint32_t>("clipIndex", 0);
    caseContext.Set<uint32_t>("frameLength", arm::app::asr::g_FrameLength);
    caseContext.Set<uint32_t>("frameStride", arm::app::asr::g_FrameStride);
    caseContext.Set<float>("scoreThreshold", arm::app::asr::g_ScoreThreshold);  /* Score threshold. */
    caseContext.Set<uint32_t>("ctxLen", arm::app::asr::g_ctxLen);  /* Left and right context length (MFCC feat vectors). */
    caseContext.Set<const std::vector <std::string>&>("labels", labels);
    caseContext.Set<arm::app::AsrClassifier&>("classifier", classifier);

    // flag to check if the specific key-word is detected e.g. hi
    bool kw_flag = false;
    caseContext.Set<bool>("kw_flag", kw_flag);

    bool executionSuccessful = true;

    static bool audio_inited;

    if (!audio_inited) {
        int err = hal_audio_init(16000);  // Initialize audio at 16,000 Hz
        if (err) {
            info("hal_audio_init failed with error: %d\n", err);
        }
        audio_inited = true;
    }

    // only in kws mode
    hal_get_audio_data(audio_inf_kws + AUDIO_SAMPLES_KWS, AUDIO_STRIDE_KWS);


    /* 
    int32_t ret;
    ret = ospi_flash_send();
    info(" FLASH send status: %ld \n", ret);
    ret = ospi_flash_read();
    info(" FLASH read status: %ld \n", ret);
    */

    
    while(1){

        info("KKK ......................\n");

    // button press mode    
    /*
    if (run_requested_())
        {   
            send_name("Dinusha"); 
            
            // hal_get_audio_data(audio_inf_asr, AUDIO_SAMPLES_ASR); // recorded audio data in mono
           
            // // Wait until the buffer is fully populated
            // err = hal_wait_for_audio();
            // if (err) {
            //     info("hal_wait_for_audio failed with error: %d\n", err);
            // }

            // hal_audio_preprocessing(audio_inf_asr, AUDIO_SAMPLES_ASR);

            // sleep_or_wait_msec(300);

            // std::vector<int16_t> audio_inf_vector(audio_inf_asr, audio_inf_asr + AUDIO_SAMPLES_ASR);
            // caseContext.Set("audio_inf_vector", audio_inf_vector);

            // info("Audio recoded......................\n");

            // executionSuccessful = ClassifyAudioHandler(
            //                         caseContext,
            //                         1,
            //                         false);

        }
        */
        

        // kws mode 
        int err = hal_wait_for_audio();
        if (err) {
            printf_err("hal_get_audio_data failed with error: %d\n", err);
        }
        info("BBB ......................\n");
        // move buffer down by one stride, clearing space at the end for the next stride
        std::copy(audio_inf_kws+ AUDIO_STRIDE_KWS, audio_inf_kws + AUDIO_STRIDE_KWS + AUDIO_SAMPLES_KWS, audio_inf_kws);
        // start receiving the next stride immediately before we start heavy processing, so as not to lose anything
        hal_get_audio_data(audio_inf_kws + AUDIO_SAMPLES_KWS, AUDIO_STRIDE_KWS);
        hal_audio_preprocessing(audio_inf_kws + AUDIO_SAMPLES_KWS - AUDIO_STRIDE_KWS, AUDIO_STRIDE_KWS);

        std::vector<int16_t> audio_inf_vector(audio_inf_kws, audio_inf_kws + AUDIO_SAMPLES_KWS + AUDIO_STRIDE_KWS);
        caseContext.Set("audio_inf_vector", audio_inf_vector);

        info("AAA ......................\n");

        // if (!caseContext.Get<bool>("kw_flag")){

            info("KWS ......................\n");

            executionSuccessful = ClassifyAudioHandler(
                                    caseContext,
                                    0,
                                    false);


        // }
        
        // Run ASR if the key-word detected
       if (caseContext.Get<bool>("kw_flag")) {

            info("ASR ......................\n");

            executionSuccessful = ClassifyAudioHandler(
                                    caseContext,
                                    1,
                                    false);

            caseContext.Set<bool>("kw_flag", false); // Reset flag 

        }
        

    }

    info("Main loop terminated.\n");
}



static bool VerifyTensorDimensions(const arm::app::Model& model)
{
    /* Populate tensor related parameters. */
    TfLiteTensor* inputTensor = model.GetInputTensor(0);
    if (!inputTensor->dims) {
        printf_err("Invalid input tensor dims\n");
        return false;
    } else if (inputTensor->dims->size < 3) {
        printf_err("Input tensor dimension should be >= 3\n");
        return false;
    }

    TfLiteTensor* outputTensor = model.GetOutputTensor(0);
    if (!outputTensor->dims) {
        printf_err("Invalid output tensor dims\n");
        return false;
    } else if (outputTensor->dims->size < 3) {
        printf_err("Output tensor dimension should be >= 3\n");
        return false;
    }

    return true;
}