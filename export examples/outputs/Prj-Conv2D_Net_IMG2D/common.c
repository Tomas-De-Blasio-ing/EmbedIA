#include "common.h"
#include "math.h"
#include <stdio.h>

#define ALLOC_BUFFER_SZ 10000

typedef struct{
    uint32_t size;
    void  * data;
} raw_buffer;


#define MAX_BUFFER 2

static unsigned char id = MAX_BUFFER-1;
static raw_buffer buffer[MAX_BUFFER] = {0};

static unsigned char pool_buffer[ALLOC_BUFFER_SZ];


/**
 * @brief Initializes the double buffer system to a known state
 *
 * Purpose: Resets the buffer allocation system before starting a new processing
 * sequence. Must be called before first use of swap_alloc() or when restarting
 * a pipeline workflow.
 *
 * Implementation details:
 * - Resets buffer index to MAX_BUFFER-1, so next swap_alloc() uses buffer 0
 * - Does not clear buffer contents (data remains until overwritten)
 * - No memory allocation/deallocation - just state reset
 * - Safe to call multiple times
 *
 * Typical usage: Call once at start of each processing pipeline or when
 * reinitializing the buffer system after completion of a workflow.
 */
void prepare_buffers(){
    id = MAX_BUFFER-1;
    buffer[0].size =0;
    buffer[1].size =0;
}

/**
 * @brief Allocates memory from a double buffer system with automatic swapping
 *
 * Purpose: Provides memory allocation for pipeline processing where each new
 * allocation automatically invalidates the previous one. Designed for scenarios
 * where you process data, generate output, and no longer need the input buffer.
 *
 * Implementation details:
 * - Uses two fixed buffers within a static pool
 * - Buffer 0 grows from left (start of pool)
 * - Buffer 1 grows from right (end of pool), with 4-byte alignment
 * - Buffers alternate on each call, maximizing available space
 * - Returns NULL if requested size would cause buffer collision
 * - No dynamic memory allocation - deterministic for MCU applications
 *
 * @param s Size in bytes to allocate
 * @return Pointer to allocated buffer, or NULL if insufficient space
 */
void * swap_alloc(uint32_t s){

    if (s!=0) // Apply 4-byte alignment (critical for MCU like Cortex-M0, optimal for M3/M4)
        s = (s + 3) & ~3;

    if ((buffer[1-id].size+s) > ALLOC_BUFFER_SZ){
        printf("Insufficient buffer size. Required %d bytes when available %d bytes.\n",
               s, ALLOC_BUFFER_SZ-buffer[1-id].size);
        return NULL;
    }

    if (++id == MAX_BUFFER){
        id = 0;
    }

    buffer[id].size = s;

    if (id==0){ //buffer from left
        buffer[id].data = pool_buffer;
    }else{
        buffer[id].data = pool_buffer+ALLOC_BUFFER_SZ-s ;
    }

    printf("*******************************\n memory allocated: %d bytes\n*******************************\n", buffer[0].size+buffer[1].size);
    return buffer[id].data;
}

/*
 * argmax()
 *  Finds the index of the largest value within a vector of data (data1d_t)
 * Parameters:
 *  data => data of type data1d_t to search for max.
 * Returns:
 *  search result - index of the maximum value
 */

uint32_t argmax(data1d_t data){
    float max = data.data[0];
    uint32_t i, pos = 0;

    for(i=1;i<data.length;i++){
        if(data.data[i]>max){
            max = data.data[i];
            pos = i;
        }
    }

    return pos;
}





