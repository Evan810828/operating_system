#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>

#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>

int myfork(int argc, char *argv[], char ppid_str[]){
	/* fork a child process */
	int status;
	pid_t pid;
	char pid_str[1024];

	pid = fork();
	if (pid < 0){
		printf("Fork error!\n");
	}
	else{
		if (pid == 0){
			/* child process */
			sprintf(pid_str, "%s->%d", ppid_str, getpid());
			/* recursion */
			int i;
			char *arg[argc];
			for(i=0;i<argc;i++){
				arg[i]=argv[i+1];
			}
			arg[argc] = NULL;
			
			if(argc==1){
				printf("Process Tree: %s\n\n", pid_str);
			}
			else if (argc!=1){
				myfork(argc-1, arg, pid_str);
			}
			execve(argv[0], argv, NULL);
		}
		else { 
			/* father process */

			/* wait for child process terminates */
			waitpid(pid, &status, WUNTRACED);

			/* check child process'  termination status */
			printf("Child process %d of parent process %d ", pid, getpid());
			if(WIFEXITED(status)){
				printf("terminated normally with exit code %d\n",WEXITSTATUS(status));
			}
			else if(WIFSIGNALED(status)){
				switch(WTERMSIG(status)){
					case 1:
						printf("is terminated by signal 1 (hung up)\n");
						break;
					case 2:
						printf("is terminated by signal 2 (interrupted)\n");
						break;
					case 3:
						printf("is terminated by signal 3 (quited)\n");
						break;
					case 4:
						printf("is terminated by signal 4 (illegal instruction)\n");
						break;
					case 5:
						printf("is terminated by signal 5 (trapped)\n");
						break;
					case 6:
						printf("is terminated by signal 6 (aborted)\n");
						break;
					case 7:
						printf("is terminated by signal 7 (bus error)\n");
						break;
					case 8:
						printf("is terminated by signal 8 (floating error)\n");
						break;
					case 9:
						printf("is terminated by signal 9 (killed)\n");
						break;
					case 11:
						printf("is terminated by signal 11 (segment fault)\n");
						break;
					case 13:
						printf("is terminated by signal 13 (piped)\n");
						break;
					case 14:
						printf("is terminated by signal 14 (alarmed)\n");
						break;
					case 15:
						printf("is terminated by signal 15 (terminated)\n");
						break;
					
				}
				printf("\n");
			}
			else if(WIFSTOPPED(status)){
				printf("stopped\n\n");
			}
			else{
				printf("continued\n\n");
			}

		}
	}
}

int main(int argc,char *argv[]){

	/* Implement the functions here */
	int i;
	char *arg[argc];
	for(i=0;i<argc-1;i++){
		arg[i]=argv[i+1];
	}
	arg[argc] = NULL;

	char init_tree[1024];
	sprintf(init_tree, "%d", getpid());

	myfork(argc-1, arg, init_tree);
	printf("Myfork process (%d) terminated normally\n", getpid());

	return 0;
}