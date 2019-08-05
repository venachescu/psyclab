/*

   psyclabGraphics

   Acts as the graphics manager controller, will handle the static functions of
   updating OpenGL

*/

#ifndef _PSYCLAB_GRAPHICS_H
#define _PSYCLAB_GRAPHICS_H

#include <vector>
#include <iostream>
#include <map>
#include <Eigen/Eigen>

#ifdef __APPLE__
#include <GLUT/glut.h>
#elif __linux
#include <GL/glut.h>
#endif
#include <X11/Xlib.h>

#include "osc_messages.h"

#include "SL.h"
#include "SL_openGL.h"
#include "SL_userGraphics.h"

namespace psyclab
{
    template <typename T>
    struct BaseGraphics
    {
        BaseGraphics(void)
        {
            graphics_ = static_cast<T*>(this);
        }

        BaseGraphics(size_t init_struct_size, size_t draw_struct_size)
        {
            graphics_ = static_cast<T*>(this);

            init_struct_size_ = init_struct_size;
            draw_struct_size_ = draw_struct_size;
        }

        static bool initUserGraphics(const char* hostname, int port)
        {
            if (graphics_ == NULL)
                graphics_ = new T;

            return graphics_->initialize(hostname, port);
        }

        static bool initUserGraphics(void)
        {
            return initUserGraphics(NULL, 0);
        }

        static void initTrialGraphics(void* buffer)
        {
            graphics_->initTrial(buffer);
            return;
        }

        static void drawTrialGraphics(void* buffer)
        {
            graphics_->drawTrial(buffer);
            return;
        }

        static T* graphics(void)
        {
             return graphics_;
        }

        protected:

            static T* graphics_;

            static size_t init_struct_size_;
            static size_t draw_struct_size_;

    };

    template <typename T> T* BaseGraphics<T>::graphics_ = NULL;
    template <typename T> size_t BaseGraphics<T>::init_struct_size_ = 0;
    template <typename T> size_t BaseGraphics<T>::draw_struct_size_ = 0;

    struct psyclabGraphics : public BaseGraphics<psyclabGraphics>
    {
        public:

            psyclabGraphics(void);
            psyclabGraphics(size_t init_struct_size, size_t draw_struct_size);

            virtual int initialize(const char* hostname, int port);

            virtual void initTrial(void* buffer) {};
            virtual void drawTrial(void* buffer) {};

            bool connect(const char* hostname, int port);

            OSCClient client;

    };

    // draw a string to a position model world coordinates
    void drawString(const char *str, float* x, float* color);
    void drawString(const char *str, Eigen::Vector3f& x, float* color);
    void drawString(const char *str, Eigen::Vector3f& x, Eigen::Vector4f& color);

    void drawSphere(Eigen::Vector3f& x, float radius, Eigen::Vector4f& color);
    void drawSphere(Eigen::Vector3f& x, float radius, Eigen::Vector4f& color, float alpha);

    void drawBox(Eigen::Vector3f& x, float scale, Eigen::Vector4f& color);
    void drawBox(Eigen::Vector3f& x, Eigen::Vector3f& scale, Eigen::Vector4f& color);

    void drawRotatingPot(Eigen::Vector3f& x, float scale, Eigen::Vector4f& color);
    void drawTriangleTest(void);
    void drawPotTest(void);

    void initLighting(void);
}

#endif

