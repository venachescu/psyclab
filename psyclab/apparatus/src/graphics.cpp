
#include "graphics.h"

#define SPHERE_SLICES 20
#define SPHERE_STACKS 20

using namespace std;
using namespace Eigen;

namespace psyclab
{
    OpenGLWPtr window;

    psyclabGraphics::psyclabGraphics(void) {}

    psyclabGraphics::psyclabGraphics(size_t init_struct_size,
            size_t draw_struct_size) :
        BaseGraphics<psyclabGraphics>(init_struct_size, draw_struct_size) {}

    int psyclabGraphics::initialize(const char* hostname, int port)
    {
        char command[30];
        char descripton[100];

        if (init_struct_size_)
        {
            sprintf(descripton, "Initialize psyclab Graphics");
            sprintf(command, "initTrial");
            addToUserGraphics(command, descripton, initTrialGraphics, init_struct_size_);

            cout << "[Debug] " << command << " added to user graphics (" << init_struct_size_ << ")" << endl;
        }

        if (draw_struct_size_)
        {
            sprintf(descripton, "Draw Trial Graphics");
            sprintf(command, "drawTrial");
            addToUserGraphics(command, descripton, drawTrialGraphics, draw_struct_size_);

            cout << "[Debug] " << command << " added to user graphics (" << draw_struct_size_ << ")" << endl;
        }

        if (port != 0)
        {
            connect(hostname, port);
        }

        return true;
    }

    bool psyclabGraphics::connect(const char* hostname, int port)
    {
        string host("localhost");
        if (hostname != NULL)
        {
            host.assign(hostname);
        }

        return client.connect(host.c_str(), port);
    }

    void drawString(const char *str, float x[3], float color[4])
    {
        void *font = GLUT_BITMAP_8_BY_13;
        glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT);
        glDisable(GL_LIGHTING);

        glColor4fv(color);
        glRasterPos3fv(x);

        while(*str)
        {
            glutBitmapCharacter(font, *str);
            ++str;
        }

        glEnable(GL_LIGHTING);
        glPopAttrib();
    }

    void drawAxis(void)
    {
        glPushAttrib(GL_LIGHTING_BIT | GL_CURRENT_BIT);
        glDisable(GL_LIGHTING);

        glBegin(GL_LINES);

        glColor3f(1.0, 1.0, 1.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(1.0, 0.0, 0.0);

        glColor3f(0.0, 1.0, 0.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.0, 1.0, 0.0);

        glColor3f(0.0, 0.0, 1.0);
        glVertex3f(0.0, 0.0, 0.0);
        glVertex3f(0.0, 0.0, 1.0);

        glEnd();

        glEnable(GL_LIGHTING);
        glPopAttrib();
        return;
    }


    void drawString(const char *str, Vector3f& x, float* color)
    {
        drawString(str, x.data(), color);
        return;
    }

    void drawString(const char *str, Vector3f& x, Vector4f& color)
    {
        drawString(str, x.data(), color.data());
        return;
    }

    void drawSphere(float* x, float radius, float* color)
    {
        glPushMatrix();

            glTranslated((GLfloat) x[0], (GLfloat) x[1], (GLfloat) x[2]);

            glColor4fv(color);
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
            glutSolidSphere(radius, SPHERE_SLICES, SPHERE_STACKS);

        glPopMatrix();
        return;
    }

    void drawSphere(Vector3f& x, float radius, Vector4f& color)
    {
        drawSphere(x.data(), radius, color.data());
        return;
    }

    void drawSphere(Vector3f& x, float radius, Vector4f& color, float alpha)
    {
        color[3] = alpha;
        drawSphere(x.data(), radius, color.data());
        return;
    }

    void drawBox(float* x, float* scale, float* color)
    {
        glPushMatrix();

            glColor4fv(color);
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);

            glScalef(scale[0], scale[1], scale[2]);
            glutSolidCube(1.0);

        glPopMatrix();
        return;
    }

    void drawBox(Vector3f& x, float scale, Vector4f& color)
    {
        Vector3f scale_;
        scale_.setConstant(scale);
        drawBox(x.data(), scale_.data(), color.data());
        return;
    }

    void drawRotatingPot(Eigen::Vector3f& x, float scale, Vector4f& color)
    {
        static float rotz = 0.0;
        static float roty = 0.0;
        static float rotx = 0.0;

        glPushMatrix();

            glTranslatef(x[1], x[2], x[3]);
            glRotatef(rotz, 0.0, 0.0, 1.0);
            glRotatef(roty, 0.0, 1.0, 0.0);
            glRotatef(rotx, 1.0, 0.0, 0.0);

            glColor4fv(color.data());
            glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color.data());
            glutSolidTeapot(scale);

        glPopMatrix();

        rotz += 0.1;
        roty += 0.2;
        rotx -= 0.4;

        return;
    }

    void drawPotTest(void)
    {
        static Vector3f x;
        x.setZero();

        static Vector4f color;
        color << 0.0, 0.0, 1.0;

        drawRotatingPot(x, 0.3, color);

        return;
    }

    void initLighting(void)
    {
        // set up light colors (ambient, diffuse, specular)
        GLfloat lightKa[] = {.2f, .2f, .2f, 1.0f};  // ambient light
        GLfloat lightKd[] = {.7f, .7f, .7f, 1.0f};  // diffuse light
        GLfloat lightKs[] = {1, 1, 1, 1};           // specular light
        glLightfv(GL_LIGHT0, GL_AMBIENT, lightKa);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, lightKd);
        glLightfv(GL_LIGHT0, GL_SPECULAR, lightKs);

        // position the light
        float lightPos[4] = {0, 0, 20, 1}; // positional light
        glLightfv(GL_LIGHT0, GL_POSITION, lightPos);

        glEnable(GL_LIGHT0);                        // MUST enable each light source after configuration

        /* Turn lighting on */
        glEnable(GL_LIGHTING);
        return;
    }

}
