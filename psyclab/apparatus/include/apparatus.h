
#include <iostream>
#include <iterator>
#include <vector>

#include <Eigen/Eigen>

#include "SL.h"
#include "SL_filters.h"

// task control
#include "SL_man.h"
#include "SL_tasks.h"
#include "SL_task_servo.h"

// data collection
#include "SL_collect_data.h"
#include "SL_unix_common.h"

#include "SL_vx_wrappers.h"
#include "SL_shared_memory.h"

#include "osc_messages.h"
//#include "udp_socket.h"

#define _DEBUG_ true


namespace psyclab
{
    template <typename T> struct ShellTask
    {
        static T* active_task(void)
        {
            return task_;
        }

        static std::vector<T*>& tasks(void)
        {
            return tasks_;
        }

        static T* setTask(std::string task_name)
        {
            for (unsigned int i=0; i<tasks().size(); ++i)
            {
                if (task_name.compare(tasks()[i]->task_name()) == 0)
                {
                    task_ = tasks()[i];
                    return task_;
                }

            }
            return NULL;
        }

        protected:

            ShellTask(void)
            {
                task_ = static_cast<T*>(this);
                tasks_.push_back(task_);
            }

            static T* task_;
            static std::vector<T*> tasks_;
    };

    template <typename T> std::vector<T*> ShellTask<T>::tasks_;
    template <typename T> T* ShellTask<T>::task_ = NULL;

    class BaseTask : public ShellTask<BaseTask>
    {
        public:

            virtual int initTrial(void)
            {
                return true;
            }

            virtual int runTrial(void)
            {
                return true;
            }

            virtual int changeTrial(void)
            {
                return false;
            }

            virtual void stopTrial(void)
            {
                return;
            }

            virtual const char* task_name(void)
            {
                return NULL;
            }

    };

    template <typename A, typename T> struct BaseApparatus
    {
        static bool initialize(const char* hostname, int port)
        {
            if (instance_ == NULL)
            {
                instance_ = new A;
            }

            // connect to the python apparatus
            if (!instance_->connect(hostname, port))
            {
                return false;
            }

            // run apparatus specific initializations
            instance_->init();

            // add all tasks to SL's task servo
            for (typename std::vector<T*>::iterator it = T::tasks().begin();
                    it != T::tasks().end(); ++it)
            {
                std::cout << "[Debug] adding task... " << (*it)->task_name() << std::endl;
                addTask((*it)->task_name(), initUserTask, runUserTask, changeUserTask);
            }

            char command[50];
            char description[250];

            sprintf(command, "debug");
            sprintf(description, "Debug Psychophysics Simulator");
            addToMan(command, description, debug);

            sprintf(command, "psyclab");
            sprintf(description, "Start Psychophysics Simulator");
            addToMan(command, description, run);

            //sprintf(initial_user_command, "go0");
            strcpy(initial_user_command, command);

            return true;
        }

        static A* instance(void)
        {
            return instance_;
        }

        static OSCClient client_;
        static OSCServer server_;

        protected:

            BaseApparatus(void)
            {
                instance_ = static_cast<A*>(this);
            }

            static void debug(void)
            {
                instance()->debug();
                return;
            }

            static void run(void)
            {
                instance()->run();
                return;
            }

            static int initUserTask(void)
            {
                return instance()->initTrial();
            }

            static int runUserTask(void)
            {
                return instance()->runTrial();
            }

            static int changeUserTask(void)
            {
                return instance()->changeTrial();
            }

            static void stopTask(void)
            {
                instance()->stopTrial();
                return;
            }

            static T* setTask(std::string& task_name)
            {

                if ((task_ = T::setTask(task_name)) == NULL)
                    return NULL;

                return task_;
            }

            static A* instance_;
            static T* task_;
    };

    template <typename A, typename T> A* BaseApparatus<A, T>::instance_ = NULL;
    template <typename A, typename T> T* BaseApparatus<A, T>::task_ = NULL;
    template <typename A, typename T> OSCClient BaseApparatus<A, T>::client_;
    template <typename A, typename T> OSCServer BaseApparatus<A, T>::server_;

    class Apparatus : public BaseApparatus<Apparatus, BaseTask>
    {
        public:

            virtual void init(void);
            bool connect(const char* hostname, int port);

            void run(void);

            // functions to be replaced that wrap around a trial
            virtual int initTrial(void);
            virtual int runTrial(void);
            virtual int changeTrial(void);

            int stopTrial(void);
            int saveData(void);

            bool goToCart(Eigen::Vector3f& x_des);
            void updateState(void);
            virtual void reset(void);

            void debug(void);
            void wait(void);

            // torque
            Eigen::VectorXf u;

            // endeffector cartesian coordinates
            Eigen::VectorXf x;
            Eigen::VectorXf xd;

            // external forces
            Eigen::VectorXf f;

            // state variables
            float t;
            Eigen::VectorXf q;
            Eigen::VectorXf qd;
            Eigen::VectorXf qdd;
            Eigen::VectorXf q0;

            int control_state;

            enum control_states {
                STATE_HOLD_POSITION = 0,
                STATE_FREE_MOVEMENT = 1,
                STATE_JOINT_CONTROL = 2,
                STATE_TORQUE_CONTROL = 3,
                STATE_ENDPOINT_CONTROL = 4
            };

            std::vector<SL_DJstate> joint_target;
            std::vector<SL_Cstate> cart_target;
            std::vector<int> cart_status;
            std::vector<double> target;

            int endeff_;
            int n_dimensions_;
            int steps_;
            float damping_;
            float time_step_;

            std::vector<Filter> filters;
    };

    class BaseTrial : public BaseTask
    {
        public:

            BaseTrial(const char* name) : BaseTask(), name_(name) {}

            ~BaseTrial(void) {};

            virtual int initTrial(void)
            {
                scd();
                return true;
            }

            virtual int runTrial(void)
            {
                return true;
            }

            virtual int changeTrial(void)
            {
                return false;
            }

            virtual void stopTrial(void)
            {
                stopcd();
                apparatus->stopTrial();
            }

            virtual const char* task_name(void) {return name_.c_str();}
            virtual void debug(void) {};

            Apparatus* apparatus;

        protected:

            std::string name_;
    };

    class OSCTask : public BaseTrial
    {
        public:

            OSCTask(const char* hostname, const char* task_name) :
                BaseTrial(task_name),
                timeout_(500), rate_(10), index_(0)
            {
                connect(hostname);
            }

            int initTrial(void)
            {
                index_ = 0;
                server_.setTimeout(timeout_);
                return true;
            }

            int runTrial(void)
            {
                void* bytes = NULL;

                if (++index_ % rate_ != 0)
                    return true;

                float time_step = apparatus->time_step_ * rate_;
                client_.send("/q", apparatus->q.data(), sizeof(float) * apparatus->q.size());
                client_.send("/qd", apparatus->qd.data(), sizeof(float) * apparatus->qd.size());
                client_.send("/step", &time_step, sizeof(float));

                //std::cout << "[UDP] Receiving input state" << std::endl;
                if ((bytes = server_.listen()) == NULL)
                {
                    //std::cout << "[Error] state timeout" << std::endl;
                    apparatus->control_state = Apparatus::STATE_HOLD_POSITION;
                }
                else if (server_.route_.compare("/q") == 0)
                {
                    memcpy(apparatus->q.data(), bytes, sizeof(float) * n_dofs);
                    apparatus->control_state = Apparatus::STATE_JOINT_CONTROL;
                }
                else if (server_.route_.compare("/x") == 0)
                {
                    memcpy(apparatus->x.data(), bytes, sizeof(float) * N_CART);
                    apparatus->control_state = Apparatus::STATE_ENDPOINT_CONTROL;
                    /*std::cout << apparatus->x.transpose() << std::endl;*/
                }
                else if (server_.route_.compare("/u") == 0)
                {
                    memcpy(apparatus->u.data(), bytes, sizeof(float) * n_dofs);
                    apparatus->control_state = Apparatus::STATE_TORQUE_CONTROL;
                    std::cout << "[Torque] " << apparatus->u.transpose() << std::endl;
                } else {
                    //std::cout << "[Null] state" << std::endl;
                    apparatus->control_state = Apparatus::STATE_HOLD_POSITION;
                }

                return true;
            }

            int serverPort(void)
            {
                return server_.osck_.boundPort();
            }

        protected:

            bool connect(const char* hostname)
            {
                server_.connect(0);
                std::cout << "[Debug] controller listening on :" << server_.osck_.boundPort() << std::endl;

                if (hostname == NULL)
                {
                    client_.connect("localhost", server_.osck_.boundPort() + 1);
                    std::cout << "[Debug] sending state to 127.0.0.1:" << server_.osck_.boundPort()+1 << std::endl;
                }
                else
                {
                    client_.connect(hostname, server_.osck_.boundPort() + 1);
                    std::cout << "[Debug] sending state to " << hostname  << ":" << server_.osck_.boundPort()+1 << std::endl;
                }

                return true;
            }

            OSCClient client_;
            OSCServer server_;

            int timeout_;
            int rate_;
            int index_;

    };

}
