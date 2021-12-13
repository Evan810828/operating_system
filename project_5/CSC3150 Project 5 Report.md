<h1>CSC3150 Project 5 Report</h1>

<p align="right"><font size=4>119010341 WU Yifan</font></p>

<h2>Running Environment</h2>

**Linux Version**

```bash
root@ubuntu:/home/wyf/Desktop/source# cat /etc/issue
Ubuntu 16.04.5 LTS \n \l
```

**GCC Version**

```bash
root@ubuntu:/home/wyf/Desktop/source# gcc --version
gcc (Ubuntu 5.4.0-6ubuntu1~16.04.12) 5.4.0 20160609
Copyright (C) 2015 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
```



<h2>Execution Guide</h2>

```bash
.
├── report.pdf 
└── source 
	├── ioc_hw5.h 
	├── main.c 
	├── Makefile 
	├── mkdev.sh 
	├── rmdev.sh
	└── test.c 
```

> 1. Run make 
>
> 2. Run dmesg to check available device number 
>
> 3. Run sudo ./mkdev.sh MAJOR MINOR to build file node (MAJOR and MINOR are the available device number checked from previous step) 
>
> 4. Run ./test to start testing 
>
> 5. Run make clean to remove the module and check the messages 
>
> 6. Run sudo ./rmdev.sh to remove the file node



<h2>Sample Output</h2>

```bash
root@ubuntu:/home/wyf/Desktop/source# make
```

![image-20211206190032595](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211206190032595.png)

```bash
root@ubuntu:/home/wyf/Desktop/source# dmesg
```

![image-20211206190520062](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211206190520062.png)

```bash
root@ubuntu:/home/wyf/Desktop/source# sudo ./mkdev.sh 243 0
```

![image-20211206190540169](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211206190540169.png)

```bash
root@ubuntu:/home/wyf/Desktop/source# ./test
```

![image-20211206190838299](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211206190838299.png)

![image-20211206190733238](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211206190733238.png)

```bash
root@ubuntu:/home/wyf/Desktop/source# make clean
```

![image-20211206191614305](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211206191614305.png)

![image-20211206191658987](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211206191658987.png)

```bash
root@ubuntu:/home/wyf/Desktop/source# sudo ./rmdev.sh
```

![image-20211206191750099](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211206191750099.png)



<h2>Program Design--Basic Task</h2>

<h4>Task Outline</h4>

![image-20211208151418765](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211208151418765.png)

In this assignment, we are required to **implement an arithmetic device and simulate a device driver in kernel mode to control this device**

1. initiate a device in kernel mode
2. allocate a block of memory to simulate registers for device
3. use ioctl functions to configure the device
4. use file operations in kernel mode to control the device
5. the device should be able to conduct some arithmetic operations

<h4>Initiate Device</h4>

First, we use `alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME)` to allocate a range of character devices. Here we only need one character device, so we simply set the `DEV_BASEMINOR` to be 0 and `DEV_COUNT` to be 1. 

The function will output the device number into the variable `dev`. Note that we will need to create a file system node for the character device, so we use this `dev` together with `MAJOR()` and `MINOR()` to get the major device number and the minor device number of our character device and output them in kernel log.

Next, we perform the following steps to initiate the device

1. use `cdev_alloc()` to allocate the character device structure, the function will return the character device structure `dev_cdev`
2. use `cdev_init()`  to initiate the device and match it with our file operations `fops`. The file operations `fops` map our designed file operation functions to the system API
3. use `cdev_add()` to add the device to the system

<h4>Allocate Registers</h4>

We design a DMA buffer to store the states of our character device.

After that, we can use `kzalloc()` to allocate a block of memory in the kernel for our DMA buffer.

For each variable in the DMA buffer, we can use `myini()`, `myins()`, and  `myinc()` to retrieve the data and  `myouts`,  `myouti`,  `myoutc` to modify the data.

![image-20211208141632369](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211208141632369.png)

<h4>ioctl Function</h4>

In our file operations structure `fops`, the IO configuration functions is named as `drv_ioctl()`. There are 2 main arguments for this function

> + configuration command `unsigned int cmd`
> + user input `unsigned long arg`

`arg` is the input from the user mode, its value equals to the address of the input variable, and we can retrieve the value of this input variable by `get_from_user()`.

`cmd` is the configuration command. In total there are 6 works for the ioctl function: `HW5_IOCSETSTUID`, `HW5_IOCSETRWOK`, `HW5_IOCSETIOCOK`, `HW5_IOCSETIRQOK`, `HW5_IOCSETBLOCK`, and `HW5_IOCWAITREADABLE`. Different command leads to different operations, so we can just use a `switch` statement to implement the structure.

1. For `HW5_IOCSETSTUID`, we retrieve the value of the input variable from user mode and print it in the kernel log
2. For `HW5_IOCSETRWOK`, `HW5_IOCSETIOCOK`, and `HW5_IOCSETIRQOK`we can just print "xx OK" 
3. For `HW5_IOCSETBLOCK`, we need to justify the value of the input variable to determine the current IO mode, and then print the IO mode in the kernel log
4.  `HW5_IOCWAITREADABLE` is for synchronizing. We should sleep the kernel until the arithmetic function completes the computaion and write back the answer.

<h4>Write Function</h4>

Firstly, we need to determine the current IO mode through the value of `Blocking/non-Blocking IO` in DMA .

Secondly, the `drv_write()` function contains an argument `char __user *buffer` which is the address of the input data from user mode. We can retrieve the input data from user mode by using `copy_from_user()` and temporarily store the data in our structure `dataIn`.

To formally store the input data, we need to transfer the value of `dataIn->a`, `dataIn->b`, and `dataIn->c` into our DMA buffer.

After that, we may inititate the work routine and add it to our system. Note that for Blocking IO we need to use `flush_scheduled_work()` to flush global work queue to block until its completion.

<h4>Arithmetic Function</h4>

Firstly, the arithmetic function needs to retrieve the input data from the DMA buffer. The input data includes the arithmetic operator and the 2 operands.

In this assignment, our arithmetic function should be able to complete 5 types of tasks inlcuding `+`, `-`, `*`, `/`, and finding-prime-number. (The detail of the arithmetic operations algorithms are omitted here.)

After complete the computation, the arithmetic function should write back the answer into the DMA buffer immediately. Furthermore, for the non-Blocking IO mode, we need to set the READABLE variable in the DMA buffer to be 1.

<h4>Read Function</h4>

Firstly, we retrieve the computation answer from the DMA buffer.

Next, the `drv_read()` function contains an argument `char __user *buffer` which is the address of the output variable from user mode. We can use `put_user()` to output our computation answer to the user mode.

We should reset the READABLE variable in DMA buffer to be 0 after the read operation.

<h4>Exit deivce</h4>

After we complete all the operations for the device, we also need to free the memory. We have allocated for the DMA buffer and the work routine, unregister the deivce we have added to the system, and free the IRQ line(bonus).



<h2>Program Design--Bonus</h2>

In the bonus task, we will try to count the number of interrupts of the keyboard.

Using the command line `watch -n 1 cat /proc/interrupts`, we can see different interrupt request lines, and we choose `IRQ_NUM` to be 1 which denotes the keyboard interrupts.

![image-20211208152911707](C:\Users\86183\AppData\Roaming\Typora\typora-user-images\image-20211208152911707.png)

<h4>Allocate Interrput line</h4>

We use `request_irq(IRQ_NUM, irq_handler, IRQF_SHARED, "my_irq", dev_cdev)` to allocate a interrupt line.

> + `irq_handler` is the function we use to count the number of interrupts
> + To enable the IRQ function, we set the 3rd argument, IRQ flags to be `IRQF_SHARED`
> + The last argument is a cookie passed to the IRQ handler to differentiate interrupts, we can use our device number as parameter

<h4>Interrupt Handler</h4>

In the DMA buffer, we have a block named as `DMACOUNTADDR`, which can be used to store the interrupt count.

Everytime the interrupt handler is invoked, we just add 1 to the value of the interrupt count stored in the DMA buffer.

<h4>Print Out Total Counts of Interrupts</h4>

When we exit the device, we can print out the total interrupt count retrieved from the DMA buffer in the kernel log.



<h2>What I have learned</h2>

Through this assignment, I have had a deeper understanding about the IO control mechanisms. The device is allocated and added to the system when the user use it. Each device has a device controller which includes multiple registers to store the device states. And the most impressive part about IO control is the 2 different IO mode. Tradditionally, we use the non-blocking IO mode, which transfer the data by character and is very slow. We also need to do synchronization between the CPU and the device. However, the Blocking IO mode is much more convenient and efficient.

I also gain deeper understanding about the data trasferring between the use mode the the kernel mode. By using `copy_from_user()`, `get_user()` and `put_user()`, I can easily transfer data among different levels. The most important in this mechanism is using the address of the user variables to retrieve data as well as write back data.

