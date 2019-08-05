#include "apparatus.h"

#include "SL_dynamics.h"
#include "SL_kinematics.h"

using namespace std;
using namespace Eigen;

namespace psyclab
{
    void Apparatus::run(void)
    {
        if (_DEBUG_)
            cout << "[Debug] Starting apparatus!" << endl;

        string command;
        string route;

        while (true)
        {
            command.clear();
            route.clear();
            if (!server_.listen(command))
            {
                // timeouts
                continue;
            }

            route.assign(server_.route_);

            if (_DEBUG_)
                cout << "[Debug] Message: " << route << "; " << command << endl;

            if (route.compare("/apparatus") == 0)
            {
                if (command.compare("break") == 0)
                {
                    break;
                }
                else if (command.compare("stop") == 0)
                {
                    stop(NULL);
                }
                else if (command.compare("reset") == 0)
                {
                    this->reset();
                }
                else if (command.compare("saveData") == 0)
                {
                    saveData();
                }
            }
            else if (route.compare("/trial") == 0)
            {
                if (command.compare("stop") == 0)
                    task_->stopTrial();
            }
            else if (route.compare("/task") == 0 && strcmp(current_task_name, NO_TASK) == 0)
            {
                if ((task_ = BaseTask::setTask(command)) == NULL)
                {
                    cerr << "[Error] couldn't set task: " << command << endl;
                    debug();
                }
                else
                {
                    ((BaseTrial*) task_)->apparatus = this;
                    if (!setTaskByName((char*) command.c_str()))
                    {
                        cerr << "[Error] task servo could not initialize the task" << endl;
                    }
                }
            }
            else if (route.compare("/command") == 0 && strcmp(current_task_name, NO_TASK) == 0)
            {
                sendCommandLineCmd((char*) command.c_str());
            }
            //wait();
        }

        return;
    }

    void Apparatus::init(void)
    {
        endeff_ = 1;
        damping_ = 0.95;
        time_step_ = 1.0 / (float) task_servo_rate;

        for (int e=0; e<=n_endeffs; ++e)
            cart_target.push_back(cart_state[e]);

        cart_status.push_back(0);
        target.push_back(0.0);

        for (int d=0; d<n_dofs; ++d)
            joint_target.push_back(joint_default_state[d+1]);

        init_filters();
        for (int d=0; d<=n_dofs; ++d)
        {
            Filter filter;
            filter.cutoff = 5;
            filters.push_back(filter);

            for (int i=0; i<3; ++i)
            {
                filters[d].raw[i] = 0.0;
                filters[d].filt[i] = 0.0;
            }
        }

        for (int e=1; e<=n_endeffs; ++e)
        {
            for (int c=1; c<=N_CART; ++c)
            {
                if (e == endeff_)
                    cart_status.push_back(1);
                else
                    cart_status.push_back(0);
            }

            for (int c=N_CART+1; c<=6; ++c)
                cart_status.push_back(0);

            for (int c=1; c<=6; ++c)
                target.push_back(0.0);
        }

        q = VectorXf::Zero(n_dofs);
        qd = VectorXf::Zero(n_dofs);
        qdd = VectorXf::Zero(n_dofs);
        u = VectorXf::Zero(n_dofs);

        x = VectorXf::Zero(N_CART);
        xd = VectorXf::Zero(N_CART);
        f = VectorXf::Zero(N_CART);

        return;
    }

    int Apparatus::initTrial(void)
    {
        updateState();
        //client_.send("/task", (void*) task_->task_name(), strlen(task_->task_name()));
        if (!task_->initTrial())
            return false;

        client_.send("/trial/start", 1);
        return true;
    }

    int Apparatus::runTrial(void)
    {
        t += time_step_;

        int step_ = task_->runTrial();
        if (step_ < 0)
        {
            task_->stopTrial();
            return step_;
        }

        switch (control_state)
        {
            case STATE_JOINT_CONTROL:

                for (int d=0; d<n_dofs; ++d)
                {
                    joint_des_state[d+1].th = q[d];
                    joint_des_state[d+1].thd = qd[d];
                    joint_des_state[d+1].thdd = 0.0;
                    joint_des_state[d+1].uff = 0.0;
                }
                break;

            case STATE_TORQUE_CONTROL:

                for (int d=0; d<n_dofs; ++d)
                {
                    joint_des_state[d+1].th = joint_state[d+1].th;
                    joint_des_state[d+1].thd = joint_state[d+1].thd * damping_;
                    joint_des_state[d+1].thdd = 0.0;
                    joint_des_state[d+1].uff = u[d];
                }
                break;

            case STATE_ENDPOINT_CONTROL:

                for (int c=0; c<N_CART; ++c)
                {
                    target[(endeff_-1)*6+c+1] =
                        (x[c] - cart_des_state[endeff_].x[c+1]) * 20. + 0.5 * xd[c];
                }

                for (int d=0; d<n_dofs; ++d)
                    joint_target[d+1].th = joint_des_state[d+1].th;

                inverseKinematics(&joint_target[0],
                        endeff,
                        joint_opt_state,
                        &target[0],
                        &cart_status[0],
                        time_step_);

                for (int d=0; d<n_dofs; ++d)
                {
                    joint_des_state[d+1].th = joint_target[d+1].th;
                    joint_des_state[d+1].thd = joint_target[d+1].thd;
                    joint_des_state[d+1].thdd = filt(
                            (joint_target[d+1].thd - joint_des_state[d+1].thd)
                            * (double) task_servo_rate, &filters[d]);
                    joint_des_state[d+1].uff = 0.0;
                }

                break;

            case STATE_FREE_MOVEMENT:

                for (int d=0; d<n_dofs; ++d)
                {
                    joint_des_state[d+1].th = joint_state[d+1].th;
                    joint_des_state[d+1].thd = joint_state[d+1].thd * damping_;
                    joint_des_state[d+1].thdd = 0.0;
                    joint_des_state[d+1].uff = 0.0;
                }

                break;

            // default is to hold current position
            default:
            case STATE_HOLD_POSITION:

                for (int d=0; d<n_dofs; ++d)
                {
                    joint_des_state[d+1].th = joint_state[d+1].th;
                    joint_des_state[d+1].thd = 0.0;
                    joint_des_state[d+1].thdd = 0.0;
                    joint_des_state[d+1].uff = 0.0;
                }

                break;
        }

        //if (state != STATE_TORQUE_CONTROL && state != STATE_HOLD_POSITION)
        if (control_state != STATE_TORQUE_CONTROL)
            SL_InvDyn(joint_state, joint_des_state, endeff, &base_state, &base_orient);

        //for (int d=0; d<n_joints_; ++d)
            //for (int c=0; c<N_CART; ++c)
                //joint_des_state[d].uff += J[c+1][d+1] * f[c];

        for (int d=0; d<n_dofs; ++d)
        {
            // check for joint limits
            if (joint_des_state[d+1].th >= joint_range[d+1][MAX_THETA])
            {
                joint_des_state[d+1].th = joint_range[d+1][MAX_THETA] - 0.01;
                joint_des_state[d+1].thd = 0.0;
                joint_des_state[d+1].thdd = 0.0;
                joint_des_state[d+1].uff = 0.0;
            }

            if (joint_des_state[d+1].th <= joint_range[d+1][MIN_THETA])
            {
                joint_des_state[d+1].th = joint_range[d+1][MIN_THETA] + 0.01;
                joint_des_state[d+1].thd = 0.0;
                joint_des_state[d+1].thdd = 0.0;
                joint_des_state[d+1].uff = 0.0;
            }

            // check for NaN
            //if (joint_des_state[d+1].uff != joint_des_state[d+1].uff)
                //joint_des_state[d+1].uff = 0.0;
        }

        return true;
    }

    int Apparatus::stopTrial(void)
    {
        setTaskByName((char*) NO_TASK);
        client_.send("/trial/stop", 1);
        return true;
    }

    int Apparatus::changeTrial(void)
    {
        if (BaseTask::active_task() == NULL)
        {
            cout << "[Debug] select a task"  << endl;
            return false;
        }
        return BaseTrial::active_task()->changeTrial();
    }

    void Apparatus::updateState(void)
    {
        for (int d=0; d<n_dofs; ++d)
        {
            q[d] = joint_state[d+1].th;
            qd[d] = joint_state[d+1].thd;
            qdd[d] = joint_state[d+1].thdd;
            u[d] = joint_state[d+1].load;
        }

        for (int c=0; c<N_CART; ++c)
        {
            x[c] = cart_state[endeff_].x[c+1];
            xd[c] = cart_state[endeff_].xd[c+1];
        }
    }

    bool Apparatus::goToCart(Eigen::Vector3f& x_des)
    {
        SL_Cstate start_state[n_endeffs+1];
        int status[2*N_CART*n_endeffs+1];

        for (int e=0; e<n_endeffs; ++e)
        {
            for (int c=1; c<=N_CART; ++c)
            {
                start_state[e+1].x[c] = x_des[c-1];
                status[2*N_CART*e+c] = 1;
            }

            for (int c=N_CART+1; c<=N_CART*2; ++c)
                status[2*N_CART*e+c] = 0;
        }

        if (!go_cart_target_wait(start_state, status, 0.75))
            return true;

        return false;
    }

    int Apparatus::saveData(void)
    {
        int n_data_file = 0;
        FILE* last_data = fopen(".last_data", "r");
        if (last_data != NULL)
        {
            int code = fscanf(last_data, "%d", &n_data_file);
            fclose(last_data);
        }

        char command[] = "saveData";
        sendCommandLineCmd(command);

        client_.send("/sl/data_file", n_data_file);
        return n_data_file;
    }

    void Apparatus::wait(void)
    {
        int pause = 10 * 1000; // 100 milliseconds
        while (strcmp(current_task_name, NO_TASK))
        {
            usleep(pause);
        }

        return;
    }

    bool Apparatus::connect(const char* hostname, int port)
    {
        string host("localhost");
        if (hostname != NULL)
        {
            host.assign(hostname);
        }

        // connect outgoing port the python apparatus manager
        client_.connect(host.c_str(), port);
        if (_DEBUG_)
            cout << "[Debug] Client connected to " << host << ":" << port << endl;

        // open listener for commands from python apparatus
        server_.connect(0);

        if (!server_.osck_.isBound())
        {
            cerr << "[Error] Server failed to connect: " << server_.osck_.errorMessage() << endl;
            return false;
        }

        if (_DEBUG_)
            cout << "[Debug] Server connected: " << server_.osck_.localHostNameWithPort() << endl;

        // send start confirmation - send port to experiment manager
        vector<int> ids;
        ids.push_back(parent_process_id);
        ids.push_back(server_.osck_.boundPort());
        client_.send("/apparatus/connect", ids);
        return true;
    }

    void Apparatus::debug(void)
    {
        cout << "[Debug] psyclab Apparatus" << endl;
        cout << "Tasks: " << endl;
        return;
    }

    void Apparatus::reset(void)
    {
        cout << "Reset apparatus" << endl;
        return;
    }

}
