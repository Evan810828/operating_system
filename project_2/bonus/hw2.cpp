#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>
#include <cmath>

#define ROW 10
#define COLUMN 50 


struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 
int log_head[9];
int log_length[9];
char map[ROW+10][COLUMN] ; 
pthread_mutex_t mutex;
// game status
int win_id;
int isQuit;
const char *str;
int isOnLog;
char last_step;
int speed;
// mouse monitor
int mouseX, mouseY;
int mouseDistance;
float sliderX;
bool mouseLeftDown;

void JudgeStatus(){
	// first check isQuit
	if(!isQuit){
		switch(map[frog.x][frog.y]){
		case ' ':
			isQuit = 1;
			str =  "You lose the game!";
		case '=':
			isOnLog = frog.x; // frog.x indicates the log index
			break;
		case '|':
			if(frog.x==0){
				isQuit = 1;
				str =  "You win the game!";
			}
			break;
		}
	}
}

void display(void){
	pthread_mutex_lock(&mutex);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
 	
	if(!isQuit){
		// slider
		glBegin(GL_LINES);
			glLineWidth(30.0);
			glColor3ub(0,0,0);	
			glVertex2f(15, 11.5);
			glVertex2f(33, 11.5);
		glEnd();
		glBegin(GL_POINTS);
		glVertex2f(sliderX, 11.5);
		// bank
		glColor3f(0.6f, 1.0f, 0.6f);
		for(int i=0; i<49; i++){
			glVertex2f(i, 0);
			glVertex2f(i, 10);
		}
		// logs
		glColor3ub(139,69,19);
		for(int x=0; x<9; x++){
			for(int y=0; y<5+log_length[x]; y++){
				glVertex2f((log_head[x]+y)%49, x+1);
			}
		}
		// frogs
		glColor3ub(0,127,0);
		glVertex2f(frog.y, frog.x);
		glEnd();
	}else{
		int n = strlen(str);
		glRasterPos2i(20,6);  
		for (int i = 0; i < n; i++)  
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *(str+i));
	}
	glutSwapBuffers();
	pthread_mutex_unlock(&mutex);
}

void process_key(unsigned char key, int x, int y){
	pthread_mutex_lock(&mutex);
	map[frog.x][frog.y] = last_step; // recover the last step
	if( key == 'w' || key == 'W' ){
		frog.x -= 1;
	}
	else if( key == 'a' || key == 'A' ){
		if(frog.y>0){
			frog.y -= 1;
		}
	}
	else if( key == 'd' || key == 'D' )	{
		if(frog.y<48){
			frog.y += 1;
		}
	}
	else if( key == 's' || key == 'S' ){
		if(frog.x<11){
			frog.x += 1;
		}
	}
	else if( key == 'q' || key == 'Q' ){
		if(!isQuit) str =  "You exit the game!";
		isQuit = 1;
	}
	// judge status
	JudgeStatus();
	if(!isQuit){
		last_step = map[frog.x][frog.y];
		map[frog.x][frog.y] = '0';
	}
	pthread_mutex_unlock(&mutex);
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
		log_head[log_index-1] = head_index;
		// if the frog is on log, move the frog as well
		if(last_step=='=' && log_index == (long)isOnLog){
			frog.y += direction;
			// then check if frog.y is out of range
			if(frog.y>48 || frog.y<0){
				isOnLog = 0;
				isQuit = 1;
				str = "You lose the game!";
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
		}
		pthread_mutex_unlock(&mutex);
		// sleep
		usleep(speed);
	}
}

void myIdle(void){
	glutPostRedisplay();
}

void myMouse(int button, int state, int x, int y){
	mouseX = x;
    mouseY = y;
	if(button==GLUT_LEFT_BUTTON)
	{
		if(state == GLUT_DOWN)
		{
			if(abs(mouseX/30-sliderX)<=1 && abs(mouseY/30-11.5)<=2) 
				mouseLeftDown = true;
		}
		else if(state == GLUT_UP)
			mouseLeftDown = false;
	}
}
void mouseMotion(int x, int y){
	if (mouseLeftDown){
        mouseDistance += (x - mouseX);
		if((float)mouseDistance/30 + 15>=15 && (float)mouseDistance/30 + 15<=33){
			sliderX = (float)mouseDistance/30 + 15;
			speed = 200000 - (float)mouseDistance/30*8000;
		}
        mouseX = x;
        mouseY = y;
		glutPostRedisplay();
    }
}

void *frog_move( void *a ){
    glutDisplayFunc(display);
	glutKeyboardFunc(process_key);
	glutMouseFunc(myMouse);
	glutMotionFunc(mouseMotion);
	glutIdleFunc(myIdle);
    glutMainLoop();
}

void initGlut( int argc, char *argv[] ){
    glutInit(&argc, argv); // initialize glut
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE); // single buffer
    glutInitWindowSize(1440, 440); // window size
    glutInitWindowPosition(200, 200); // window position
    win_id = glutCreateWindow("test"); // create a window
    glOrtho(-0.5f, 48.5f, 12.5f, -0.5f, 20.0f, -1.0f);// set coordinate base
	glClearColor(0.54f, 0.824f, 1.0f, 1.0f);// river
    glPointSize(30); // set point size
}

int main( int argc, char *argv[] ){
	// initialize the game status
	argc = argc;
	argv = argv;
	isQuit = 0;
	isOnLog = 0;
	last_step = '|';
	sliderX = 15;
	speed = 200000;
	for(int i=0; i<9; i++){
		log_length[i] = rand()%10; // randomly generate the length of logs
	}
	initGlut(argc, argv);

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x+1][frog.y] = '0' ; 

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
