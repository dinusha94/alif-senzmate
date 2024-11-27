#ifndef EXT_LOADER_H
#define EXT_LOADER_H

/**************************************************************************//**
 * @file     Flash.hpp
 * @author   Dinusha Nuwan
 * @email    dinusha@senzmate.com
 * @version  V1.0.0
 * @date     22-11-2024
 * @brief    Class for external flash memory operations 
 * @bug      None.
 * @Note     None
 ******************************************************************************/

#include <vector>
#include <string>
#include <cstring>  // for memcpy
#include <cstdint>
#include "log_macros.h"    
#include "ospi_flash.h"


int32_t erase_flash(void){

    int32_t ret;
    ret = ptrDrvFlash->EraseChip();
    return ret;
}

int32_t flash_send_model(uint32_t address,  uint16_t *write_buff, uint32_t length) 
{
    int32_t ret;

    // uint16_t write_buff_local[length];
    // std::memcpy(write_buff_local, write_buff, length);

    // Address 0x00,  subsector 0 
    ret = ptrDrvFlash->ProgramData(address, write_buff, length);

    ARM_FLASH_STATUS flash_status;
    do {
        flash_status = ptrDrvFlash->GetStatus();
    } while (flash_status.busy);

    return ret;
}


int32_t flash_read_model(uint32_t address, uint32_t length)
{
    int32_t ret;
    uint16_t read_buff[length];

    ret = ptrDrvFlash->ReadData(address, read_buff, length);

    ARM_FLASH_STATUS flash_status;
    do {
        flash_status = ptrDrvFlash->GetStatus();
    } while (flash_status.busy);

    // std::copy(read_buff + 4, read_buff + length, read_buff);

    printf("Read buf:\n");
    for (size_t i = 0; i < length; ++i) { 
        printf("0x%04x, ",  read_buff[i]);
    }
    printf("\n");

    return ret;
}

#endif 