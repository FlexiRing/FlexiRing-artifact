#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"

#ifdef __cplusplus
extern "C" {
#endif

extern void *phUART;

#define CHECK_ERRORS(x)                                                       \
    if ((x) != AM_HAL_STATUS_SUCCESS)                                         \
    {                                                                         \
        error_handler(x);                                                     \
    }
extern volatile uint32_t ui32LastError;
    
void error_handler(uint32_t ui32ErrorStatus);
extern const am_hal_uart_config_t g_sUartConfig;
void am_uart_isr(void);
void uart_print(char *pcStr);
    
void print_init(void);
void test();
#ifdef __cplusplus
}
#endif
