/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/micro/micro_time.h"

// Set in micro/tools/make/targets/cortex_m_generic_makefile.inc.
// Needed for the DWT and PMU counters.
#include "RTE_Components.h"
#include CMSIS_device_header
#include "core_cm4.h"

// Ambiq HAL clock macros
#include "am_mcu_apollo.h"

namespace tflite {

uint32_t ticks_per_second() { return AM_HAL_CLKGEN_FREQ_MAX_HZ; }

uint32_t GetCurrentTimeTicks() {
    // Use DWT cycle counter as high-resolution timer.
    static bool is_initialized = false;

    if (!is_initialized) {
        CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  // Enable trace subsystem (gates DWT).

        // Reset and enable cycle counter.
        DWT->CYCCNT = 0;
        DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;

        is_initialized = true;
    }

    return DWT->CYCCNT;
}

}  // namespace tflite
