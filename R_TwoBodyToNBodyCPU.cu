// Name: Dillon Gatlin
// Two body problem
// nvcc R_TwoBodyToNBodyCPU.cu -o temp -lglut -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

/*
 What to do:
 This is some crude code that moves two bodies around in a box, attracted by gravity and 
 repelled when they hit each other. Take this from a two-body problem to an N-body problem, where 
 NUMBER_OF_SPHERES is a #define that you can change. Also clean it up a bit so it is more user friendly.
*/

/*
 Purpose:
 To learn about Nbody code.
*/

// Include files
#include <GL/glut.h>
#include <GL/glu.h>
#include <GL/gl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Defines
#define XWindowSize 1000
#define YWindowSize 1000
#define STOP_TIME 10000.0
#define DT        0.0001
#define GRAVITY 0.1 
#define MASS 10.0  	
#define DIAMETER 1.0
#define SPHERE_PUSH_BACK_STRENGTH 50.0
#define PUSH_BACK_REDUCTION 0.1
#define DAMP 0.01
#define DRAW 100
#define LENGTH_OF_BOX 6.0
#define MAX_VELOCITY 5.0
//simulate more n-bodies
#define NUMBER_OF_SPHERES 100

// Globals
const float XMax = (LENGTH_OF_BOX/2.0);
const float YMax = (LENGTH_OF_BOX/2.0);
const float ZMax = (LENGTH_OF_BOX/2.0);
const float XMin = -(LENGTH_OF_BOX/2.0);
const float YMin = -(LENGTH_OF_BOX/2.0);
const float ZMin = -(LENGTH_OF_BOX/2.0);
/*
float px1, py1, pz1, vx1, vy1, vz1, fx1, fy1, fz1, mass1; 
float px2, py2, pz2, vx2, vy2, vz2, fx2, fy2, fz2, mass2;*/
//-----------------------------------------------------------we changed these to arrays to handle multiple spheres
float px[NUMBER_OF_SPHERES], py[NUMBER_OF_SPHERES], pz[NUMBER_OF_SPHERES]; 
float vx[NUMBER_OF_SPHERES], vy[NUMBER_OF_SPHERES], vz[NUMBER_OF_SPHERES]; 
float fx[NUMBER_OF_SPHERES], fy[NUMBER_OF_SPHERES], fz[NUMBER_OF_SPHERES], mass[NUMBER_OF_SPHERES];

// Function prototypes
void set_initail_conditions();
void Drawwirebox();
void draw_picture();
void keep_in_box();
void get_forces();
void move_bodies(float);
void nbody();
void Display(void);
void reshape(int, int);
int main(int, char**);

//augment to hanld multiple spheres (the arrays)
void set_initail_conditions()
{ 
	time_t t;
	srand((unsigned) time(&t));
	int yeahBuddy, i, j;
	float dx, dy, dz, seperation;

	for (i = 0; i < NUMBER_OF_SPHERES; i++) 
    {
        //yeah buddy will decide if the generated spot overlaps with another sphere
        yeahBuddy = 0;
        // Keep generating until this sphere doesn't overlap with any previous one
        while (yeahBuddy == 0)
        {
            px[i] = (LENGTH_OF_BOX - DIAMETER) * rand() / (float)RAND_MAX - (LENGTH_OF_BOX - DIAMETER) / 2.0f;
            py[i] = (LENGTH_OF_BOX - DIAMETER) * rand() / (float)RAND_MAX - (LENGTH_OF_BOX - DIAMETER) / 2.0f;
            pz[i] = (LENGTH_OF_BOX - DIAMETER) * rand() / (float)RAND_MAX - (LENGTH_OF_BOX - DIAMETER) / 2.0f;

            yeahBuddy = 1; 
            //Check against all previously placed spheres
            for (j = 0; j < i; j++)
            {
                dx = px[i] - px[j];
                dy = py[i] - py[j];
                dz = pz[i] - pz[j];
                seperation = sqrtf(dx*dx + dy*dy + dz*dz);

                if (seperation < DIAMETER) 
                {
                    yeahBuddy = 0; 
                    break;
                }
            }
        }
    
	vx[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	vy[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	vz[i] = 2.0*MAX_VELOCITY*rand()/RAND_MAX - MAX_VELOCITY;
	mass[i] = 1.0;
    }
}

void Drawwirebox()
{		
	glColor3f (5.0,1.0,1.0);
	glBegin(GL_LINE_STRIP);
		glVertex3f(XMax,YMax,ZMax);
		glVertex3f(XMax,YMax,ZMin);	
		glVertex3f(XMax,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMax);
		glVertex3f(XMax,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		
		glVertex3f(XMin,YMax,ZMax);
		glVertex3f(XMin,YMax,ZMin);	
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMin,YMax,ZMax);	
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMax);
		glVertex3f(XMax,YMin,ZMax);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMin,ZMin);
		glVertex3f(XMax,YMin,ZMin);		
	glEnd();
	
	glBegin(GL_LINES);
		glVertex3f(XMin,YMax,ZMin);
		glVertex3f(XMax,YMax,ZMin);		
	glEnd();
	
}
//convert draw picture to handle the arrays
void draw_picture()
{
	float radius = DIAMETER/2.0;
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	Drawwirebox();
	
	for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
		glColor3d((float)i / NUMBER_OF_SPHERES, 1.0 - (float)i / NUMBER_OF_SPHERES, 0.5);
		glPushMatrix();
		glTranslatef(px[i], py[i], pz[i]);
		glutSolidSphere(radius, 20, 20);
		glPopMatrix();
	}
	glutSwapBuffers();
}
//convert if statements to handle the arrays
void keep_in_box()
{
	float halfBoxLength = (LENGTH_OF_BOX - DIAMETER)/2.0;
	for (int i = 0; i < NUMBER_OF_SPHERES; i++) {
        if(px[i] > halfBoxLength)
        {
            px[i] = 2.0*halfBoxLength - px[i];
            vx[i] = - vx[i];
        }
        else if(px[i] < -halfBoxLength)
        {
            px[i] = -2.0*halfBoxLength - px[i];
            vx[i] = - vx[i];
        }
        
        if(py[i] > halfBoxLength)
        {
            py[i] = 2.0*halfBoxLength - py[i];
            vy[i] = - vy[i];
        }
        else if(py[i] < -halfBoxLength)
        {
            py[i] = -2.0*halfBoxLength - py[i];
            vy[i] = - vy[i];
        }
                
        if(pz[i] > halfBoxLength)
        {
            pz[i] = 2.0*halfBoxLength - pz[i];
            vz[i] = - vz[i];
        }
        else if(pz[i] < -halfBoxLength)
        {
            pz[i] = -2.0*halfBoxLength - pz[i];
            vz[i] = - vz[i];
        }
    }
	
}
//convert get forces to handle arrays
void get_forces()
{
	float dx,dy,dz,r,r2,dvx,dvy,dvz,forceMag,inout;


    // Reset forces before summation
    for (int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        fx[i] = 0.0f;
        fy[i] = 0.0f;
        fz[i] = 0.0f;
    }

    for (int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        for (int j = i + 1; j < NUMBER_OF_SPHERES; j++)
        {
            dx = px[j] - px[i];
            dy = py[j] - py[i];
            dz = pz[j] - pz[i];

            r2 = dx * dx + dy * dy + dz * dz;
            r = sqrt(r2);
            //added to avoid div by 0, skip force calculation if r=0
            //diving by zero would cause program to crash
            if (r == 0.0f) continue; 

            forceMag = mass[i] * mass[j] * GRAVITY / r2;

            // Collision 
            if (r < DIAMETER)
            {
                dvx = vx[j] - vx[i];
                dvy = vy[j] - vy[i];
                dvz = vz[j] - vz[i];
                inout = dx * dvx + dy * dvy + dz * dvz;
                if (inout <= 0.0f)
                    forceMag += SPHERE_PUSH_BACK_STRENGTH * (r - DIAMETER);
                else
                    forceMag += PUSH_BACK_REDUCTION * SPHERE_PUSH_BACK_STRENGTH * (r - DIAMETER);
            }

                
                fx[i] += forceMag * dx / r;
                fy[i] += forceMag * dy / r;
                fz[i] += forceMag * dz / r;

                fx[j] -= forceMag * dx / r;
                fy[j] -= forceMag * dy / r;
                fz[j] -= forceMag * dz / r;
        }
    }
}

//convert move bodies to handle arrays
void move_bodies(float time)
{
	 for (int i = 0; i < NUMBER_OF_SPHERES; i++)
    {
        if (time == 0.0)
        {
            vx[i] += 0.5 * DT * (fx[i] - DAMP * vx[i]) / mass[i];
            vy[i] += 0.5 * DT * (fy[i] - DAMP * vy[i]) / mass[i];
            vz[i] += 0.5 * DT * (fz[i] - DAMP * vz[i]) / mass[i];
        }
        else
        {
            vx[i] += DT * (fx[i] - DAMP * vx[i]) / mass[i];
            vy[i] += DT * (fy[i] - DAMP * vy[i]) / mass[i];
            vz[i] += DT * (fz[i] - DAMP * vz[i]) / mass[i];
        }

        //Update positions
        px[i] += DT * vx[i];
        py[i] += DT * vy[i];
        pz[i] += DT * vz[i];
    }
	
	keep_in_box();
}

void nbody()
{	
	int    tdraw = 0;
	float  time = 0.0;

	set_initail_conditions();
	
	draw_picture();
	
	while(time < STOP_TIME)
	{
		get_forces();
	
		move_bodies(time);
	
		tdraw++;
		if(tdraw == DRAW) 
		{
			draw_picture(); 
			tdraw = 0;
		}
		
		time += DT;
	}
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glutSwapBuffers();
	glFlush();
	nbody();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);

	glMatrixMode(GL_PROJECTION);

	glLoadIdentity();

	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, 50.0);

	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("N Body 3D");

	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);

	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);

	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}

