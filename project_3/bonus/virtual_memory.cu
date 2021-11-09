#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>


__device__ void init_invert_page_table(VirtualMemory *vm) {

  for (int i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
    vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] = 0; // initialize LRU
  }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer; //physical memory = 32KB
  vm->storage = storage; // disk = 128KB
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE; // 16K := 2^14
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE; // 32K := 2^15
  vm->STORAGE_SIZE = STORAGE_SIZE; // 128K := 2^17
  vm->PAGE_ENTRIES = PAGE_ENTRIES; // 1024

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
    int virtual_page_number = addr / vm->PAGESIZE;
    int page_offset = addr % vm->PAGESIZE;
    // data is in the page table & physical memory
    for (int physical_frame_number = 0; physical_frame_number < vm->PAGE_ENTRIES; physical_frame_number++) {
      if (vm->invert_page_table[physical_frame_number] == 0x00000000) { // not an empty entry
        if (vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES] == virtual_page_number
        &&  vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 3] == threadIdx.x
        ) { // found in page table
          for(int i=0; i<vm->PAGE_ENTRIES; i++){
            if(vm->invert_page_table[i]==0x00000000){
              vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1;
            }
          }
          vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 2] = 0; // set LRU bit
          return vm->buffer[physical_frame_number*vm->PAGESIZE + page_offset];
        }
      }
    }
    *vm->pagefault_num_ptr += 1; // a page fault
    // if there is still an empty entry
    for (int physical_frame_number = 0; physical_frame_number < vm->PAGE_ENTRIES; physical_frame_number++) {
      if (vm->invert_page_table[physical_frame_number] == 0x80000000) { // an empty entry
        vm->invert_page_table[physical_frame_number] = 0x00000000;
        vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES] = virtual_page_number;
        vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 2] = 0; // set LRU bit
        vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 3] = threadIdx.x;
        return NULL;
      }
    }
    // else it is in the disk, swapping using LRU
    int max_clock_time = 0;
    int least_recent_entry = 0;
    for (int physical_frame_number = 0; physical_frame_number < vm->PAGE_ENTRIES; physical_frame_number++) {
       if (vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 2] > max_clock_time) {
           max_clock_time = vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 2];
           least_recent_entry = physical_frame_number;
       }
    }
    // put the victim into the disk and put the target data back to the physical memory
    for(int i=0; i<vm->PAGESIZE; i++){
      vm->storage[vm->invert_page_table[least_recent_entry + vm->PAGE_ENTRIES] * vm->PAGESIZE + i] = vm->buffer[least_recent_entry * vm->PAGESIZE + i];
      vm->buffer[least_recent_entry * vm->PAGESIZE + i] = vm->storage[virtual_page_number * vm->PAGESIZE + i];
    }
    // update the page table
    vm->invert_page_table[least_recent_entry + vm->PAGE_ENTRIES] = virtual_page_number; // set VPN
    for(int i=0; i<vm->PAGE_ENTRIES; i++){
      if(vm->invert_page_table[i]==0x00000000){
        vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1;
      }
    }
    vm->invert_page_table[least_recent_entry + vm->PAGE_ENTRIES * 2] = 0; // set LRU bit
    vm->invert_page_table[least_recent_entry + vm->PAGE_ENTRIES * 3] = threadIdx.x; // set pid

    return vm->buffer[least_recent_entry * vm->PAGESIZE + page_offset];
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
    int virtual_page_number = addr / vm->PAGESIZE;
    int page_offset = addr % vm->PAGESIZE;
    for (int physical_frame_number = 0; physical_frame_number < vm->PAGE_ENTRIES; physical_frame_number++) {
        if (vm->invert_page_table[physical_frame_number] == 0x00000000) { // not an empty entry
            if (vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES] == virtual_page_number
            &&  vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 3] == threadIdx.x) { // found in page table
                // update the page table
                for(int i=0; i<vm->PAGE_ENTRIES; i++){
                  if(vm->invert_page_table[i]==0x00000000){
                    vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1;
                  }
                }
                vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 2] = 0; // set LRU bit
                // write into the physical memory
                vm->buffer[physical_frame_number * vm->PAGESIZE + page_offset] = value;
                return;
            }
        }
    }
    *(vm->pagefault_num_ptr) += 1; // a page fault
    // look for invalid entries
    for (int physical_frame_number = 0; physical_frame_number < vm->PAGE_ENTRIES; physical_frame_number++) {
       if (vm->invert_page_table[physical_frame_number] == 0x80000000) { // find an empty entry
           // update the page table
           vm->invert_page_table[physical_frame_number] = 0x00000000; // set valid-bit to be valid
           vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES] = virtual_page_number; // set VPN
           for(int i=0; i<vm->PAGE_ENTRIES; i++){
              if(vm->invert_page_table[i]==0x00000000){
                vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1;
              }
            }
           vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 2] = 0; // set LRU bit // set LRU bit
           vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 3] = threadIdx.x; // set pid
           // write into the physical memory
           vm->buffer[physical_frame_number * vm->PAGESIZE + page_offset] = value;
           return;
       }
    }
    // else need swapping, using LRU
    int max_clock_time = 0;
    int least_recent_entry = 0;
    for (int physical_frame_number = 0; physical_frame_number < vm->PAGE_ENTRIES; physical_frame_number++) {
       if (vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 2] > max_clock_time) {
           max_clock_time = vm->invert_page_table[physical_frame_number + vm->PAGE_ENTRIES * 2];
           least_recent_entry = physical_frame_number;
       }
    }
    // put the victim into the disk
    for(int i=0; i<vm->PAGESIZE; i++){
      vm->storage[vm->invert_page_table[least_recent_entry + vm->PAGE_ENTRIES] * vm->PAGESIZE + i] = vm->buffer[least_recent_entry * vm->PAGESIZE + i];
    }
    // update the page table
    vm->invert_page_table[least_recent_entry + vm->PAGE_ENTRIES] = virtual_page_number; // set VPN
    for(int i=0; i<vm->PAGE_ENTRIES; i++){
      if(vm->invert_page_table[i]==0x00000000){
        vm->invert_page_table[i + vm->PAGE_ENTRIES * 2] += 1;
      }
    }
    vm->invert_page_table[least_recent_entry + vm->PAGE_ENTRIES * 2] = 0; // set LRU bit
    vm->invert_page_table[least_recent_entry + vm->PAGE_ENTRIES * 3] = threadIdx.x; // set pid
    // write into the physical memory
    vm->buffer[least_recent_entry * vm->PAGESIZE + page_offset] = value;
    return;
}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset, int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data to result buffer */
  for (int i=0; i<input_size; i++){
    results[i] = vm_read(vm, i+offset);
  }
}

