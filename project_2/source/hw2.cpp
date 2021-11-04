#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50 


struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 


char map[ROW+10][COLUMN] ; 
pthread_mutex_t mutex;
// game status
int isQuit;
int isOnLog;
char last_step;

// Determine a dirboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}

void JudgeStatus(){
	// first check isQuit
	if(isQuit){
		printf("\033[H\033[2J");
		printf("You exit the game!\n");
	}else{
		switch(map[frog.x][frog.y]){
		case ' ':
			isQuit = 1;
			printf("\033[H\033[2J");
			printf("You lose the game!\n");
			break;
		case '=':
			isOnLog = frog.x; // frog.x indicates the log index
			break;
		case '|':
			if(frog.x==0){
				isQuit = 1;
				printf("\033[H\033[2J");
				printf("You win the game!\n");
			}
			break;
		}
	}
	
}

void *logs_move( void *t ){
	long log_index = (long)t;
	log_index++; // logs start from row index 1
	long head_index = rand()%48; // decides the position of the log
	int direction = (log_index % 2)*2 -1; // odd row moves right

	while(!isQuit){
		pthread_mutex_lock(&mutex);
		// move the log
		if(direction==1){
			head_index = (head_index + direction)%(COLUMN-1); // moves right
		}else if(direction==-1){
			head_index = COLUMN-1+((head_index + direction)%(COLUMN-1)); // moves left
		}
		// if the frog is on log, move the frog as well
		if(last_step=='=' && log_index == (long)isOnLog){
			frog.y += direction;
			// then check if frog.y is out of range
			if(frog.y>48 || frog.y<0){
				isOnLog = 0;
				isQuit = 1;
				printf("\033[H\033[2J");
				printf("You lose the game!\n");
			}
		}
		if(!isQuit){
			for(int i=0; i<15; i++){
				if((head_index+i)%(COLUMN-1)==frog.y && log_index==frog.x){
					map[log_index][(head_index+i)%(COLUMN-1)] = '0';
				}else{
					map[log_index][(head_index+i)%(COLUMN-1)] = '=';
				}
			}
			// recover the map
			if(direction==-1){
				map[log_index][(head_index+15)%(COLUMN-1)] = ' ';
			}else if(direction==1){
				map[log_index][(head_index-1)%(COLUMN-1)+1] = ' ';
			}
			// print
			printf("\033[H\033[2J");
			for(int i = 0; i <= ROW; ++i)	
				puts( map[i] );
		}
		pthread_mutex_unlock(&mutex);
		// sleep
		usleep(50000);
	}
}

void *frog_move( void *a ){
	while (!isQuit){
		if( kbhit() ){
			pthread_mutex_lock(&mutex);
			char dir = getchar() ; 
			map[frog.x][frog.y] = last_step; // recover the last step
			if( dir == 'w' || dir == 'W' ){
				frog.x -= 1;
			}
			else if( dir == 'a' || dir == 'A' ){
				if(frog.y>0){
					frog.y -= 1;
				}
			}
			else if( dir == 'd' || dir == 'D' )	{
				if(frog.y<48){
					frog.y += 1;
				}
			}
			else if( dir == 's' || dir == 'S' ){
				if(frog.x<10){
					frog.x += 1;
				}
			}
			else if( dir == 'q' || dir == 'Q' ){
				isQuit= 1;
			}
			// judge status
			JudgeStatus();
			if(!isQuit){
				last_step = map[frog.x][frog.y];
				map[frog.x][frog.y] = '0';
			}
			pthread_mutex_unlock(&mutex);
		}
	}
}

int main( int argc, char *argv[] ){
	// initialize the game status
	isQuit = 0;
	isOnLog = 0;
	last_step='|';

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 

	//Print the map into screen
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );


	/*  Create pthreads for wood move and frog control.  */
	pthread_t frog_thread;
	pthread_t log_thread[9];
	pthread_mutex_init(&mutex, NULL);
	
	pthread_create(&frog_thread, NULL, frog_move, NULL); // frog
	for(long log_index=0; log_index<9; log_index++){
		pthread_create(&log_thread[log_index], NULL, logs_move, (void*)log_index); // logs
	}

	pthread_join(frog_thread, NULL);
	for(long log_index=0; log_index<9; log_index++){
		pthread_join(log_thread[log_index], NULL);
	}

	pthread_mutex_destroy(&mutex);
	pthread_exit(NULL);

	return 0;

}
