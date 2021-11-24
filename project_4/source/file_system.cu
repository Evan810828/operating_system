#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ __managed__ u32 gtime = 0;


__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}



__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
  // 1. check whether the file exists
  int isFound=0;
  for(u32 FCB_index=0; FCB_index<fs->FCB_ENTRIES; FCB_index++){
    isFound=1;
    int i=0;
    while(s[i]!='\0' || fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + i]!='\0'){
      if(s[i]!=fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + i]){
        isFound=0;
        break;
      }
      i++;
    }
    if(isFound){
      if(op==G_READ){
        return FCB_index; // return read pointer
      }
      if(op==G_WRITE){
        fs_gsys(fs, RM, s); // remove the original file
        break;
      }
    }
  }
  // 2. create a new file for G_WRITE
  for(u32 FCB_index=0; FCB_index<fs->FCB_ENTRIES; FCB_index++){ // search for an empty FCB
    if(fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE]==0){ // no filename
      // filename
      int i=0;
      while(s[i]!='\0'){
        if(i==fs->MAX_FILENAME_SIZE) break;
        fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + i] = s[i];
        i++;
      }
      // file pointer (starting block)
      for(u32 block_index=0; block_index<32768; block_index++){ // search for an empty block
        uint8_t bit = (fs->volume[block_index/8]<<(block_index%8))&0x80;
        if(bit==0x00){
          *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] = block_index;
          break;
        }
      }
      // file size
      *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] = 0;
      // modified time
      *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 8] = gtime;

      return FCB_index; // return write pointer
    }
  }
}


__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
  u32 starting_block = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + fp*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
  for(int i=0; i<size; i++){
    output[i] = fs->volume[fs->FILE_BASE_ADDRESS + starting_block*fs->STORAGE_BLOCK_SIZE + i];
  }
}

__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
  gtime++;
  char filename[20];
  for(int i=0; i<20; i++){
    if(fs->volume[fs->SUPERBLOCK_SIZE + fp*fs->FCB_SIZE + i]=='\0'){
      filename[i] = '\0';
    }else{
      filename[i] = fs->volume[fs->SUPERBLOCK_SIZE + fp*fs->FCB_SIZE + i];
    }
  }
  u32 new_fp = fs_open(fs, filename, G_WRITE);
  u32 starting_block = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + new_fp*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
  // write data
  for(int i=0;i<size;i++){
    fs->volume[fs->FILE_BASE_ADDRESS + starting_block*fs->STORAGE_BLOCK_SIZE + i] = input[i];
  }
  // update bit map
  for(int i=0;i<=size/32;i++){
    fs->volume[(starting_block + i)/8] = fs->volume[(starting_block + i)/8] | (0x80>>((starting_block+i)%8)); // 0 -> 1
  }
  // update FCB
  *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + new_fp*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4] = size;
  *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + new_fp*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 8] = gtime;
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
  if(op==LS_D){
    printf("===sorted by modified time===\n");
    int time_array[1024];
    int count=0;
    for(u32 i=0; i<fs->FCB_ENTRIES; i++){
      if(fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE]!=0){
        time_array[i] = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 8];
        count++;
      }else{
        time_array[i] = -1;
      }
    }
    for(u32 i=0; i<count; i++){
      u32 max_time_entry = 0;
      for(u32 j=0; j<1024; j++){
        if(time_array[j] >= time_array[max_time_entry]){
          max_time_entry = j;
        }
      }
      char filename[20];
      for(int j=0;j<fs->MAX_FILENAME_SIZE;j++){
        if(fs->volume[fs->SUPERBLOCK_SIZE + max_time_entry*fs->FCB_SIZE + j]!=0){
          filename[j] = fs->volume[fs->SUPERBLOCK_SIZE + max_time_entry*fs->FCB_SIZE + j];
        }else{
          filename[j] = '\0';
        }
      }
      printf("%s\n", filename);
      time_array[max_time_entry] = 0;
    }
  }else if(op==LS_S){
    printf("===sorted by size===\n");
    int size_array[1024];
    int count=0;
    for(u32 i=0; i<fs->FCB_ENTRIES; i++){
      if(fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE]!=0){
        size_array[i] = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
        count++;
      }else{
        size_array[i] = -1;
      }
    }
    for(u32 i=0; i<count; i++){
      u32 max_size_entry = 0;
      for(u32 j=0; j<1024; j++){
        if(size_array[j] >= size_array[max_size_entry]){
          max_size_entry = j;
        }
      }
      char filename[20];
      for(int j=0;j<fs->MAX_FILENAME_SIZE;j++){
        if(fs->volume[fs->SUPERBLOCK_SIZE + max_size_entry*fs->FCB_SIZE + j]!=0){
          filename[j] = fs->volume[fs->SUPERBLOCK_SIZE + max_size_entry*fs->FCB_SIZE + j];
        }else{
          filename[j] = '\0';
        }
      }
      printf("%s %d\n", filename, size_array[max_size_entry]);
      size_array[max_size_entry] = 0;
    }
  }
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
  // 1. check whether the file exists
  int isFound=0;
  u32 FCB_index=0;
  for(; FCB_index<fs->FCB_ENTRIES; FCB_index++){
    isFound=1;
    int i=0;
    while(s[i]!='\0' || fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + i]!='\0'){
      if(s[i]!=fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + i]){
        isFound=0;
        break;
      }
      i++;
    }
    if(isFound){
      break;
    }
  }
  if(!isFound) return;
  // 2. delete data
  u32 starting_block = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
  u32 size = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE + 4];
  // clear up file contents
  if(size!=0){
    for(int i=0;i<size;i++){
      fs->volume[fs->FILE_BASE_ADDRESS + starting_block*fs->STORAGE_BLOCK_SIZE + i] = 0;
    }
    // update bit map
    for(int i=0;i<=size/32;i++){
      fs->volume[(starting_block + i)/8] = fs->volume[(starting_block + i)/8] & (uint8_t)~(0x80>>((starting_block+i)%8)); // 0 -> 1
    }
  }
  // clear up FCB
  for(int i=0; i<fs->FCB_SIZE; i++){
    fs->volume[fs->SUPERBLOCK_SIZE + FCB_index*fs->FCB_SIZE + i] = 0;
  }
  // 3. compaction
  int has_next=0;
  u32 next_FCB_index=0;
  for(int i=FCB_index; i<fs->FCB_ENTRIES; i++){
    if(fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE]!=0){
      next_FCB_index = i;
      has_next = 1;
      break;
    }
  }
  if(!has_next) return;
  // compact file contents
  u32 next_block = *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + next_FCB_index*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE];
  u32 range = 0;
  while(next_block+range<=32768){
    uint8_t bit = (fs->volume[(next_block+range)/8]<<((next_block+range)%8))&0x80;
    if(bit==0x00) break;
    for(int i=0;i<fs->STORAGE_BLOCK_SIZE;i++){
      // move data
      fs->volume[fs->FILE_BASE_ADDRESS + (starting_block+range)*fs->STORAGE_BLOCK_SIZE + i] = fs->volume[fs->FILE_BASE_ADDRESS + (next_block+range)*fs->STORAGE_BLOCK_SIZE + i];
      fs->volume[fs->FILE_BASE_ADDRESS + (next_block+range)*fs->STORAGE_BLOCK_SIZE + i] = 0;
    }
    // update bit map
    fs->volume[(starting_block+range)/8] = fs->volume[(starting_block+range)/8] | (0x80>>((starting_block+range)%8)); // 0 -> 1
    fs->volume[(next_block+range)/8] = fs->volume[(next_block+range)/8] & (uint8_t)(~(0x80>>((starting_block+range)%8))); // 1 -> 0
    range++;
  }
  // update file pointer
  for(int i=0; i<fs->FCB_ENTRIES; i++){
    if(fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE]!=0){
      if(*(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE]>starting_block && *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE]!=0x80000000){
        *(u32*)&fs->volume[fs->SUPERBLOCK_SIZE + i*fs->FCB_SIZE + fs->MAX_FILENAME_SIZE] -= (next_block-starting_block);
      }
    }
  }
}
