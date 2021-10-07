#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){

	int status;
	pid_t pid;

	/* fork a child process */
	printf("Process start to fork\n");
	pid = fork();

	if (pid < 0){
		printf("Fork error!\n");
	}
	else{
		if (pid == 0){
			/* execute test program */ 
            int i;
            char *arg[argc];
            
            for(i=0;i<argc-1;i++){
                arg[i]=argv[i+1];
            }
            arg[argc-1]=NULL;

			printf("I'm the Child Process, my pid = %d\n", getpid());
			printf("Child process start to execute test program:\n");
			execve(arg[0],arg,NULL);
			
			perror("execve");
			exit(EXIT_FAILURE);
		}
		else {
			printf("I'm the Parent Process, my pid = %d\n", getpid());

			/* wait for child process terminates */
            waitpid(pid, &status, WUNTRACED);
            printf("Parent process receives the signal\n");

			/* check child process'  termination status */
            if(WIFEXITED(status)){
                printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));
            }
            else if(WIFSIGNALED(status)){
				switch(WTERMSIG(status)){
					case 1:
						printf("child process is hung up\n");
						break;
					case 2:
						printf("child process is interrupted\n");
						break;
					case 3:
						printf("child process quited\n");
						break;
					case 4:
						printf("child process had an illegal instruction\n");
						break;
					case 5:
						printf("child process is trapped\n");
						break;
					case 6:
						printf("child process is aborted\n");
						break;
					case 7:
						printf("child process had a bus error\n");
						break;
					case 8:
						printf("child process had a floating error\n");
						break;
					case 9:
						printf("child process is killed\n");
						break;
					case 11:
						printf("child process had a segment fault\n");
						break;
					case 13:
						printf("child process is piped\n");
						break;
					case 14:
						printf("child process is alarmed\n");
						break;
					case 15:
						printf("child process is terminated\n");
						break;
					
				}
                printf("CHILD EXECUTION FAILED\n");
            }
            else if(WIFSTOPPED(status)){
				printf("child process stopped\n");
                printf("CHILD PROCESS STOPPED\n");
            }
            else{
                printf("CHILD PROCESS CONTINUED\n");
            }
            exit(0);
		}
	}
}