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



int32_t flash_send_model(uint32_t address,  uint8_t *write_buff, uint32_t length) 
{
    int32_t ret;
    uint8_t write_buff_local[length];

    std::memcpy(write_buff_local, write_buff, length);

    // Address 0x00,  subsector 0 
    ret = ptrDrvFlash->ProgramData(address, write_buff_local, length);

    ARM_FLASH_STATUS flash_status;
    do {
        flash_status = ptrDrvFlash->GetStatus();
        info("busy \n");
    } while (flash_status.busy);

    return ret;
}


int32_t flash_read_model()
{
    int32_t ret;
    uint32_t count = 0, iter = 0;
    uint8_t read_buff[64];

    ret = ptrDrvFlash->ReadData(0xC0000000, read_buff, 64);

    ARM_FLASH_STATUS flash_status;
    do {
        flash_status = ptrDrvFlash->GetStatus();
        info("busy \n");
    } while (flash_status.busy);

    printf("Data in read_buff:\n");
    for (int i = 0; i < 64; ++i) {
        printf("%d ", read_buff[i]);  // %04X prints 4-digit hexadecimal with leading zeros
        if ((i + 1) % 8 == 0) {
            printf("\n");  // Newline every 8 elements
        }

    }
    printf("\n");

    return ret;
}




#endif 