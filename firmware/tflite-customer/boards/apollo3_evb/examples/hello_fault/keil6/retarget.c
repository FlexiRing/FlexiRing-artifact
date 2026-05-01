#include <stdio.h>

typedef int FILEHANDLE;

// Disable semihosting 
// Note:
//   Use microlib will disable semihosting
//   If not, disable semihosting using folllow code
//#pragma import(__use_no_semihosting_swi)  // ARM Compiler 5
__asm(".global __use_no_semihosting\n\t");  // ARM Compiler 6 



void _ttywrch(int ch)
{
ch = ch;
}

int ferror(FILE *f)
{
(void)f;
return EOF;
}

void _sys_exit(int return_code)
{
(void)return_code;
while (1) {
};
}

FILEHANDLE _sys_open(const char *name, int openmode)
{
return 1;
}

int _sys_close(FILEHANDLE fh)
{
return 0;
}

int _sys_write(FILEHANDLE fh, const unsigned char *buf, unsigned len, int mode)
{
//your_device_write(buf, len);
return 0;
}

int _sys_read(FILEHANDLE fh, unsigned char *buf, unsigned len, int mode)
{
return -1;
}

int _sys_istty(FILEHANDLE fh)
{
return 0;
}

int _sys_seek(FILEHANDLE fh, long pos)
{
return -1;
}

long _sys_flen(FILEHANDLE fh)
{
return -1;
}