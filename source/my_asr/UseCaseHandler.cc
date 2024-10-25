/*
 * SPDX-FileCopyrightText: Copyright 2021-2022 Arm Limited and/or its affiliates
 * <open-source-office@arm.com> SPDX-License-Identifier: Apache-2.0
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
#include "UseCaseHandler.hpp"

#include "AsrClassifier.hpp"
#include "AsrResult.hpp"
#include "AudioUtils.hpp"
#include "ImageUtils.hpp"
#include "InputFiles.hpp"
#include "OutputDecode.hpp"
#include "UseCaseCommonUtils.hpp"
#include "Wav2LetterModel.hpp"
#include "Wav2LetterPostprocess.hpp"
#include "Wav2LetterPreprocess.hpp"
#include "hal.h"
#include "log_macros.h"

#include <vector>


// extern uint32_t m55_comms_handle;
// m55_data_payload_t mhu_data;

// using arm::app::ApplicationContext;


#define AUDIO_SAMPLES 47712 // samples/sec, 1sec sample
#define AUDIO_STRIDE 16000 // 0.5 seconds

static int16_t audio_inf[AUDIO_SAMPLES + AUDIO_STRIDE];


namespace arm {
namespace app {

    // namespace audio {
    // using namespace arm::app::audio;
    // }

    /**
     * @brief       Presents ASR inference results.
     * @param[in]   results   Vector of ASR classification results to be displayed.
     * @return      true if successful, false otherwise.
     **/
    static bool PresentInferenceResult(const std::vector<asr::AsrResult>& results);

    // static std::string last_name;

    // static void send_name(arm::app::kws::KwsResult &result)
    // {
        
    //     mhu_data.id = 3; // id for senzmate app
    //     if (result.m_resultVec.empty()) {
    //         last_name.clear();
    //         return;
    //     }

    //     arm::app::ClassificationResult classification = result.m_resultVec[0];

    //     if (classification.m_label != last_name) {
    //         if (classification.m_label == "go" || classification.m_label == "stop") {
    //             info("******************* send_name, FOUND \"%s\", copy data end send! ******************\n", classification.m_label.c_str());
    //             strcpy(mhu_data.msg, classification.m_label.c_str());
    //             __DMB();
    //             SERVICES_send_msg(m55_comms_handle, &mhu_data);
    //         }
    //         last_name = classification.m_label;
    //     }

    // }

    /* ASR inference handler. */
    bool ClassifyAudioHandler(ApplicationContext& ctx, uint32_t clipIndex, bool runAll)
    {
        auto& model          = ctx.Get<Model&>("model");
        auto& profiler       = ctx.Get<Profiler&>("profiler");
        auto mfccFrameLen    = ctx.Get<uint32_t>("frameLength");
        auto mfccFrameStride = ctx.Get<uint32_t>("frameStride");
        auto scoreThreshold  = ctx.Get<float>("scoreThreshold");
        auto inputCtxLen     = ctx.Get<uint32_t>("ctxLen");     

        /* If the request has a valid size, set the audio index. */
        // if (clipIndex < NUMBER_OF_FILES) {
        //     if (!SetAppCtxIfmIdx(ctx, clipIndex, "clipIndex")) {
        //         return false;
        //     }
        // }
        // auto initialClipIdx                    = ctx.Get<uint32_t>("clipIndex");
        // constexpr uint32_t dataPsnTxtInfStartX = 20;
        // constexpr uint32_t dataPsnTxtInfStartY = 40;

        if (!model.IsInited()) {
            printf_err("Model is not initialised! Terminating processing.\n");
            return false;
        }

        TfLiteTensor* inputTensor  = model.GetInputTensor(0);
        TfLiteTensor* outputTensor = model.GetOutputTensor(0);

        /* Get input shape. Dimensions of the tensor should have been verified by
         * the callee. */
        TfLiteIntArray* inputShape = model.GetInputShape(0);

        const uint32_t inputRowsSize = inputShape->data[Wav2LetterModel::ms_inputRowsIdx];
        const uint32_t inputInnerLen = inputRowsSize - (2 * inputCtxLen);

        /* Audio data stride corresponds to inputInnerLen feature vectors. */
        const uint32_t audioDataWindowLen = (inputRowsSize - 1) * mfccFrameStride + (mfccFrameLen);
        const uint32_t audioDataWindowStride = inputInnerLen * mfccFrameStride;

        info(" audioDataWindowLen : %ld \n", audioDataWindowLen);
        info(" audioDataWindowStride : %ld \n", audioDataWindowStride);

        /* NOTE: This is only used for time stamp calculation. */
        const float secondsPerSample = (1.0 / audio::Wav2LetterMFCC::ms_defaultSamplingFreq);

        /* Set up pre and post-processing objects. */
        AsrPreProcess preProcess = AsrPreProcess(inputTensor,
                                                 Wav2LetterModel::ms_numMfccFeatures,
                                                 inputShape->data[Wav2LetterModel::ms_inputRowsIdx],
                                                 mfccFrameLen,
                                                 mfccFrameStride);

        std::vector<ClassificationResult> singleInfResult;
        const uint32_t outputCtxLen = AsrPostProcess::GetOutputContextLen(model, inputCtxLen);
        AsrPostProcess postProcess  = AsrPostProcess(outputTensor,
                                                    ctx.Get<AsrClassifier&>("classifier"),
                                                    ctx.Get<std::vector<std::string>&>("labels"),
                                                    singleInfResult,
                                                    outputCtxLen,
                                                    Wav2LetterModel::ms_blankTokenIdx,
                                                    Wav2LetterModel::ms_outputRowsIdx);

        static bool audio_inited;
        if (!audio_inited) {
            int err = hal_audio_init(16000);
            if (err) {
                printf_err("hal_audio_init failed with error: %d\n", err);
                return false;
            }
            audio_inited = true;
        }

        // Start first fill of final stride section of buffer
        hal_get_audio_data(audio_inf + AUDIO_SAMPLES, AUDIO_STRIDE);


        /* Loop to process audio clips. */
        do {
             // Wait until stride buffer is full - initiated above or by previous interation of loop
            int err = hal_wait_for_audio();
            if (err) {
                printf_err("hal_get_audio_data failed with error: %d\n", err);
                return false;
            }

            // move buffer down by one stride, clearing space at the end for the next stride
            std::copy(audio_inf + AUDIO_STRIDE, audio_inf + AUDIO_STRIDE + AUDIO_SAMPLES, audio_inf);

            // start receiving the next stride immediately before we start heavy processing, so as not to lose anything
            hal_get_audio_data(audio_inf + AUDIO_SAMPLES, AUDIO_STRIDE);

            hal_audio_preprocessing(audio_inf + AUDIO_SAMPLES - AUDIO_STRIDE, AUDIO_STRIDE);

            // const int16_t* inferenceWindow = audio_inf;

            // hal_lcd_clear(COLOR_BLACK);

            /* Get current audio clip index. */
            // auto currentIndex = ctx.Get<uint32_t>("clipIndex");

            /* Get the current audio buffer and respective size. */
            const int16_t* audioArr     =  audio_inf;  // GetAudioArray(currentIndex);
            const uint32_t audioArrSize =  AUDIO_SAMPLES;     // GetAudioArraySize(currentIndex);

            if (!audioArr) {
                printf_err("Invalid audio array pointer.\n");
                return false;
            }

            /* Audio clip needs enough samples to produce at least 1 MFCC feature. */
            if (audioArrSize < mfccFrameLen) {
                info("Not enough audio samples, minimum needed is %" PRIu32 "\n",
                           mfccFrameLen);
                return false;
            }

            /* Creating a sliding window through the whole audio clip. */
            auto audioDataSlider = audio::FractionalSlidingWindow<const int16_t>(
                audioArr, audioArrSize, audioDataWindowLen, audioDataWindowStride);

            /* Declare a container for final results. */
            std::vector<asr::AsrResult> finalResults;

            /* Display message on the LCD - inference running. */
            // std::string str_inf{"Running inference... "};
            // hal_lcd_display_text(
            //     str_inf.c_str(), str_inf.size(), dataPsnTxtInfStartX, dataPsnTxtInfStartY, 0);

            // info("Running inference on audio clip %" PRIu32 " => %s\n",
            //      currentIndex,
            //      GetFilename(currentIndex));

            size_t inferenceWindowLen = audioDataWindowLen;

            /* Start sliding through audio clip. */
            while (audioDataSlider.HasNext()) {

                /* If not enough audio, see how much can be sent for processing. */
                size_t nextStartIndex = audioDataSlider.NextWindowStartIndex();
                if (nextStartIndex + audioDataWindowLen > audioArrSize) {
                    inferenceWindowLen = audioArrSize - nextStartIndex;
                }

                const int16_t* inferenceWindow = audioDataSlider.Next();

                info("Inference %zu/%zu\n",
                     audioDataSlider.Index() + 1,
                     static_cast<size_t>(ceilf(audioDataSlider.FractionalTotalStrides() + 1)));

                /* Run the pre-processing, inference and post-processing. */
                if (!preProcess.DoPreProcess(inferenceWindow, inferenceWindowLen)) {
                    printf_err("Pre-processing failed.");
                    return false;
                }

                if (!RunInference(model, profiler)) {
                    printf_err("Inference failed.");
                    return false;
                }

                /* Post processing needs to know if we are on the last audio window. */
                postProcess.m_lastIteration = !audioDataSlider.HasNext();
                if (!postProcess.DoPostProcess()) {
                    printf_err("Post-processing failed.");
                    return false;
                }

                /* Add results from this window to our final results vector. */
                finalResults.emplace_back(asr::AsrResult(
                    singleInfResult,
                    (audioDataSlider.Index() * secondsPerSample * audioDataWindowStride),
                    audioDataSlider.Index(),
                    scoreThreshold));

// #if VERIFY_TEST_OUTPUT
//                 armDumpTensor(outputTensor,
//                               outputTensor->dims->data[Wav2LetterModel::ms_outputColsIdx]);
// #endif        /* VERIFY_TEST_OUTPUT */
            } /* while (audioDataSlider.HasNext()) */

            /* Erase. */
            // str_inf = std::string(str_inf.size(), ' ');
            // hal_lcd_display_text(
            //     str_inf.c_str(), str_inf.size(), dataPsnTxtInfStartX, dataPsnTxtInfStartY, 0);

            ctx.Set<std::vector<asr::AsrResult>>("results", finalResults);

            if (!PresentInferenceResult(finalResults)) {
                return false;
            }

            // profiler.PrintProfilingResult();

            // IncrementAppCtxIfmIdx(ctx, "clipIndex");
        // ++index;
        } while (runAll); // && ctx.Get<uint32_t>("clipIndex") != initialClipIdx

        return true;
    }

    static bool PresentInferenceResult(const std::vector<asr::AsrResult>& results)
    {
        constexpr uint32_t dataPsnTxtStartX1 = 20;
        constexpr uint32_t dataPsnTxtStartY1 = 60;
        constexpr bool allow_multiple_lines  = true;

        // hal_lcd_set_text_color(COLOR_GREEN);

        // info("Final results:\n");
        // info("Total number of inferences: %zu\n", results.size());
        /* Results from multiple inferences should be combined before processing. */
        std::vector<ClassificationResult> combinedResults;
        for (const auto& result : results) {
            combinedResults.insert(
                combinedResults.end(), result.m_resultVec.begin(), result.m_resultVec.end());
        }

        /* Get each inference result string using the decoder. */
        for (const auto& result : results) {
            std::string infResultStr = audio::asr::DecodeOutput(result.m_resultVec);

            // info("For timestamp: %f (inference #: %" PRIu32 "); label: %s\n",
            //      result.m_timeStamp,
            //      result.m_inferenceNumber,
            //      infResultStr.c_str());
        }

        /* Get the decoded result for the combined result. */
        std::string finalResultStr = audio::asr::DecodeOutput(combinedResults);

        // hal_lcd_display_text(finalResultStr.c_str(),
        //                      finalResultStr.size(),
        //                      dataPsnTxtStartX1,
        //                      dataPsnTxtStartY1,
        //                      allow_multiple_lines);

        info("Complete recognition: %s\n", finalResultStr.c_str());
        return true;
    }

} /* namespace app */
} /* namespace arm */