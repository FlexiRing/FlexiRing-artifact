#include "am_mcu_apollo.h"
#include "am_bsp.h"
#include "am_util.h"
#include "am_devices_led.h"
#include "am_hal_pwrctrl.h"
#include "am_hal_cachectrl.h"
#include "am_devices_button.h"

#include <stdio.h>
#include <stdint.h>

#include "uart_print.h"
#include "classifier.h"

//#include "gesture_model_quantized.h"
#include "gesture_model.h"
#include "gesture_input_data.h"

__asm(".global __ARM_use_no_argv\n\t");
__asm(".global __use_no_semihosting\n\t");

float input_data[180*6]={0};
float output_data[OUTPUT_LENGTH]={0};
float result[NUM_CLASSES]={0};

static void UartClearBuffers(void)
{
    uint8_t dummy[AM_HAL_UART_FIFO_MAX];
    uint32_t bytes = 0;

    do
    {
        bytes = 0;
        CHECK_ERRORS(am_hal_uart_fifo_read(phUART, dummy, sizeof(dummy), &bytes));
    }
    while (bytes != 0);

    CHECK_ERRORS(am_hal_uart_tx_flush(phUART));
}

static void RunGestureOnceAndReport(void)
{
    int status = RunGestureInference(input_data, output_data);
    
    memset(result, 0, sizeof(result));
    int ans = classify_gesture(output_data, result);
    
    //for(int i=0; i<256; i++) am_util_debug_printf("%d %.3f\r\n",i, input_data[i]);
    
    am_util_debug_printf("%d\r\n", status);
    
    for(int i = 0;i < NUM_CLASSES; i++)
    {
        am_util_debug_printf("%d %f\r\n",i, result[i]);
    }

    UartClearBuffers();
}

void read_from_csv(void)
{
    char c;
    uint32_t idx = 0;
    float val = 0.0f;
    uint32_t val_idx = 0;
    bool decimal_found = false;
    float decimal_divisor = 10.0f;
    bool first_line = true;
    float negative_flag = 1.0f;
    
    while (1)
    {
        uint32_t got = 0;

        while (got == 0)
        {
            CHECK_ERRORS(am_hal_uart_fifo_read(phUART, (uint8_t *)&c, 1, &got));
        }
        if(first_line)
        {

            if (c == '\n')
            {
                first_line = false;
            }
            continue;
        }

        if (c == ',' || c == '\n')
        {
            input_data[idx++] = val * negative_flag;
            // am_util_debug_printf("[%d]\r\n", val_idx++);
            val = 0.0f;
            negative_flag = 1.0f;
            decimal_found = false;
            decimal_divisor = 10.0f;

            if (c == 'T' || idx >= 180*6)
            {
                break; 
            }
        }
        else if (c == '-')
        {
            negative_flag = -1.0f;
        }
        else if (c >= '0' && c <= '9')
        {
            if (decimal_found)
            {
                val += (c - '0') / decimal_divisor;
                decimal_divisor *= 10.0f;
            }
            else
            {
                val = val * 10.0f + (c - '0');
            }
        }
        else if (c == '.')
        {
            decimal_found = true;
        }
    }

    RunGestureOnceAndReport();
}

int
main(void)
{
    // Max system clock
    am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);
    
    // Enable all memory (SRAM + others)
    am_hal_pwrctrl_memory_enable(AM_HAL_PWRCTRL_MEM_ALL);

    // Enable cache
    am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
    am_hal_cachectrl_enable();

    am_devices_led_array_init(am_bsp_psLEDs, AM_BSP_NUM_LEDS);
    am_devices_led_array_out(am_bsp_psLEDs, AM_BSP_NUM_LEDS, 0);
    am_hal_gpio_state_write(AM_BSP_GPIO_LED0, AM_HAL_GPIO_OUTPUT_SET);

 
    am_hal_gpio_pinconfig(AM_BSP_GPIO_BUTTON0, g_AM_HAL_GPIO_INPUT_PULLUP);

    //
    // Initialize the printf interface for ITM/SWO output.
    //
 //   am_bsp_low_power_init();
    am_bsp_itm_printf_enable();
    am_util_debug_printf("hello world \r\n");
    print_init();

    am_util_stdio_printf("hello world11 \r\n");
    if (InitGestureModel() != kTfLiteOk) {
        am_util_debug_printf("Model Init Failed!\n");
        while(1);
    }
    
    for(int i = 0; i < g_gesture_test_input_num; i++)
    {
        input_data[i] = g_gesture_test_input[i];
    }
    
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

    //RunGestureOnceAndReport();

    uint8_t rx;
    uint32_t got;
    uint32_t sent;
    while (1)
    {
        got = 0;
        while (got == 0)
        {
            CHECK_ERRORS(am_hal_uart_fifo_read(phUART, &rx, 1, &got));
        }
        if(rx == 'S')
        {
            read_from_csv();
        }
    }

}
