#include "MobileFacenet.hpp"

#include "log_macros.h"

const tflite::MicroOpResolver& arm::app::MobileFacenet::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::MobileFacenet::EnlistOperations()
{
#ifndef ETHOS_U_NPU_ASSUMED
    this->m_opResolver.AddDepthwiseConv2D();
    this->m_opResolver.AddConv2D();
    this->m_opResolver.AddAdd();
    this->m_opResolver.AddResizeNearestNeighbor();
    /*These are needed for UT to work, not needed on FVP */
    this->m_opResolver.AddPad();
    this->m_opResolver.AddMaxPool2D();
    this->m_opResolver.AddConcatenation();
#endif

    if (kTfLiteOk == this->m_opResolver.AddEthosU()) {
        info("Added %s support to op resolver\n",
            tflite::GetString_ETHOSU());
    } else {
        printf_err("Failed to add Arm NPU support to op resolver.");
        return false;
    }
    return true;
}
