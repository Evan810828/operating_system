#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include <linux/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"

// irq
#define IRQ_NUM 1 // interrupt line

// DEVICE
#define DEV_NAME "mydev"        // name for alloc_chrdev_region
#define DEV_BASEMINOR 0         // baseminor for alloc_chrdev_region
#define DEV_COUNT 1             // count for alloc_chrdev_region
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdev;

// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2
void *dma_buf;

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );

// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

// in and out function
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work_routine;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;

// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);

// irq handler
static irqreturn_t irq_handler(int irq, void *dev_id);


// Input and output data from/to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}


static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    	printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}
static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    	printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement read operation for your device */
	int ans = myini(DMAANSADDR);
	put_user(ans, (int *)buffer);
	printk("%s:%s(): ans=%d\n", PREFIX_TITLE, __func__,ans);
	myouti(0, DMAREADABLEADDR);
	return 0;
}
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement write operation for your device */
	int IOMode = myini(DMABLOCKADDR);
	char* data = kmalloc(sizeof(char) * ss, GFP_KERNEL);
	copy_from_user(data, buffer, ss);
	dataIn = (struct DataIn *) data;
	myoutc(dataIn->a, DMAOPCODEADDR);
	myouti(dataIn->b, DMAOPERANDBADDR);
	myouts(dataIn->c, DMAOPERANDCADDR);
	printk("%s:%s(): queue work\n", PREFIX_TITLE, __func__);

	INIT_WORK(work_routine, drv_arithmetic_routine);

	// Decide io mode
	if(IOMode){
		// Blocking IO
		printk("%s:%s(): block\n", PREFIX_TITLE, __func__);
		schedule_work(work_routine);
		flush_scheduled_work();
	}else{
		// Non-locking IO
		schedule_work(work_routine);
	}

	return 0;
}
static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	/* Implement ioctl setting for your device */
	int variable;
	get_user(variable, (int *)arg);
	switch (cmd)
	{
	case HW5_IOCSETSTUID:
		myouti(variable, DMASTUIDADDR);
		printk("%s,%s(): My STUID is = %d\n", PREFIX_TITLE, __func__, variable);
		break;
	case HW5_IOCSETRWOK:
		myouti(variable, DMARWOKADDR);
		printk("%s,%s(): RW OK\n", PREFIX_TITLE, __func__);
		break;
	case HW5_IOCSETIOCOK:
		myouti(variable, DMAIOCOKADDR);
		printk("%s,%s(): IOC OK\n", PREFIX_TITLE, __func__);
		break;
	case HW5_IOCSETIRQOK:
		myouti(variable, DMAIRQOKADDR);
		printk("%s,%s(): IRQ OK\n", PREFIX_TITLE, __func__);
		break;
	case HW5_IOCSETBLOCK:
		myouti(variable, DMABLOCKADDR);
		if(variable==0) printk("%s,%s(): Non-Blocking IO\n", PREFIX_TITLE, __func__);
		if(variable==1) printk("%s,%s(): Blocking IO\n", PREFIX_TITLE, __func__);
		break;
	case HW5_IOCWAITREADABLE:
		printk("%s,%s(): wait readable 1\n", PREFIX_TITLE, __func__);
		while(myini(DMAREADABLEADDR)==0){
			msleep(2000);
		}
		put_user(myini(DMAREADABLEADDR), (int *)arg);
		break;
	default:
		break;
	}
	return 0;
}

int prime(int base, short nth)
{
    int fnd=0;
    int i, num, isPrime;

    num = base;
    while(fnd != nth) {
        isPrime=1;
        num++;
        for(i=2;i<=num/2;i++) {
            if(num%i == 0) {
                isPrime=0;
                break;
            }
        }
        
        if(isPrime) {
            fnd++;
        }
    }
    return num;
}

static void drv_arithmetic_routine(struct work_struct* ws) {
	/* Implement arthemetic routine */
    int ans;

    char a = myinc(DMAOPCODEADDR);
    int b = myini(DMAOPERANDBADDR);
    int c = myins(DMAOPERANDCADDR);
	// computation
    switch(a) {
        case '+':
            ans=b+c;
            break;
        case '-':
            ans=b-c;
            break;
        case '*':
            ans=b*c;
            break;
        case '/':
            ans=b/c;
            break;
        case 'p':
            ans = prime(b, c);
            break;
        default:
            ans=0;
    }
	// write back ans
	myouti(ans, DMAANSADDR);
	// update readable
	if(myini(DMABLOCKADDR)==0){ // Non-Blocking IO
		myouti(1, DMAREADABLEADDR);
	}
	printk("%s,%s(): %d %c %d = %d\n", PREFIX_TITLE,__func__,b,a,c,ans);
}

static irqreturn_t irq_handler(int irq, void *dev_id){
	int itr_num = myini(DMACOUNTADDR) + 1;
	myouti(itr_num, DMACOUNTADDR);
	return IRQ_HANDLED;
}

static int __init init_modules(void) {
    
	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);

	/* Register chrdev */ 
	dev_t dev;
	int ret = 0;

	ret = alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME);
	if(ret)
	{
		printk("Cannot register chrdev\n");
		return ret;
	}
	
	dev_major = MAJOR(dev);
	dev_minor = MINOR(dev);
	printk("%s:%s():register chrdev(%d,%d)\n",PREFIX_TITLE,__func__,dev_major,dev_minor);

	/* Init cdev and make it alive */
	dev_cdev = cdev_alloc();

	cdev_init(dev_cdev, &fops);
	dev_cdev->owner = THIS_MODULE;
	ret = cdev_add(dev_cdev, dev, 1);
	if(ret < 0)
	{
		printk("Add chrdev failed\n");
		return ret;
	}

    /* Allocate IRQ */
    ret = request_irq(IRQ_NUM, irq_handler, IRQF_SHARED, "my_irq", dev_cdev);
    if(ret < 0)
    {
        printk("Request IRQ failed\n");
        return ret;
    }
	printk("%s:%s():request_irq %d returns %d\n",PREFIX_TITLE,__func__, IRQ_NUM,ret);

	/* Allocate DMA buffer */
	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
	printk("%s:%s():allocate dma buffer\n",PREFIX_TITLE,__func__);

	myouti(0, DMACOUNTADDR);

	/* Allocate work routine */
	work_routine = kmalloc(sizeof(typeof(*work_routine)), GFP_KERNEL);


	return 0;
}

static void __exit exit_modules(void) {

	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s:%s():free dma buffer\n",PREFIX_TITLE,__func__);

	/* Delete character device */
	unregister_chrdev_region(MKDEV(dev_major, dev_minor), DEV_COUNT);
	cdev_del(dev_cdev);
	printk("%s:%s():unregister chrdev\n",PREFIX_TITLE,__func__);

	/* Free work routine */
	kfree(work_routine);

    /* Free IRQ */
    free_irq(IRQ_NUM, dev_cdev);
	printk("%s:%s(): interrupt count = %d\n", PREFIX_TITLE, __func__, myini(DMACOUNTADDR));

	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);