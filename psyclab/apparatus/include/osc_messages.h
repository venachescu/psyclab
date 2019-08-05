

#include "udp.hh"
#include "oscpkt.hh"

#include <iostream>
#include <vector>

class OSCClient {

    public:

        OSCClient() {}

        bool connect(const char* host, int port)
        {
            osck_.connectTo(host, port);
            if (!osck_.isOk())
            {
                return false;
            }

            return true;
        }

        bool send(const char* path, void* data, int size)
        {
            oscpkt::Message msg;
            msg.init(path);
            msg.pushBlob(data, size);

            oscpkt::PacketWriter writer;
            writer.addMessage(msg);
            return osck_.sendPacket(writer.packetData(), writer.packetSize());
        }

        bool send(const char* path, int value)
        {
            oscpkt::Message msg;
            msg.init(path);
            msg.pushInt32(value);

            oscpkt::PacketWriter writer;
            writer.addMessage(msg);
            return osck_.sendPacket(writer.packetData(), writer.packetSize());
        }

        bool send(const char* path, std::vector<int>& value)
        {
            oscpkt::Message msg;
            msg.init(path);
            for (unsigned int i=0; i<value.size(); ++i)
                msg.pushInt32(value[i]);

            oscpkt::PacketWriter writer;
            writer.addMessage(msg);
            return osck_.sendPacket(writer.packetData(), writer.packetSize());
        }

        bool send(const char* path, float value)
        {
            oscpkt::Message msg;
            msg.init(path);
            msg.pushFloat(value);

            oscpkt::PacketWriter writer;
            writer.addMessage(msg);
            return osck_.sendPacket(writer.packetData(), writer.packetSize());
        }

        bool send(const char* path, std::string& value)
        {
            oscpkt::Message msg;
            msg.init(path);
            msg.pushStr(value);

            oscpkt::PacketWriter writer;
            writer.addMessage(msg);
            return osck_.sendPacket(writer.packetData(), writer.packetSize());
        }

        oscpkt::UdpSocket osck_;

    protected:

        size_t size_;
};

template <typename T> class OSCSender : OSCClient {

    public:

        OSCSender() : OSCClient()
        {
            size_ = sizeof(T);
        }

        bool send(const char* path, T& data)
        {
            return send(path, (void*) &data, sizeof(T));
        }

        oscpkt::UdpSocket osck_;

    protected:

        size_t size_;
        std::vector<T> messages_;
};


class OSCServer
{
    public:

        OSCServer(void)
        {
            timeout_ = 10000;
        }

        bool connect(int port)
        {
            osck_.bindTo(port, oscpkt::UdpSocket::OPTION_FORCE_IPV4);
            if (!osck_.isOk())
            {
                return false;
            }

            return true;
        }

        int timeout(void)
        {
            return timeout_;
        }

        bool setTimeout(int timeout)
        {
            if (timeout < 0)
                return false;

            timeout_ = timeout;
            return true;
        }

        void* listen(void)
        {
            if (osck_.isOk() && osck_.receiveNextPacket(timeout_))
            {
                oscpkt::PacketReader reader;
                reader.init(osck_.packetData(), osck_.packetSize());
                oscpkt::Message* msg;
                if  (reader.isOk() && (msg = reader.popMessage()) != 0)
                {
                    msg->arg().popBlob(buffer_);
                    route_.assign(msg->addressPattern());
                    return buffer_.data();
                }
            } else {
                std::cout << "[OSC Error] timeout (" << timeout_ << ")" << std::endl;
            }
            return NULL;
        }

        void* listen(const char* route)
        {
            while (osck_.isOk()) {
                if (osck_.receiveNextPacket(timeout_))
                {
                    oscpkt::PacketReader reader;
                    reader.init(osck_.packetData(), osck_.packetSize());
                    oscpkt::Message* msg;
                    if  (reader.isOk() && (msg = reader.popMessage()) != 0)
                    {
                        if (msg->partialMatch(route).popBlob(buffer_).isOkNoMoreArgs())
                        {
                            route_.assign(msg->addressPattern());
                            return buffer_.data();
                        }
                    }
                } else {
                    std::cout << "[OSC Error] timeout (" << timeout_ << ")" << std::endl;
                }
            }

            return NULL;
        }

        bool listen(const char* route, void* ptr)
        {
            if (osck_.isOk() && osck_.receiveNextPacket(timeout_))
            {
                oscpkt::PacketReader reader;
                reader.init(osck_.packetData(), osck_.packetSize());
                oscpkt::Message* msg;
                if  (reader.isOk() && (msg = reader.popMessage()) != 0)
                {
                    if (msg->partialMatch(route).popBlob(buffer_))
                    {
                        route_.assign(msg->addressPattern());
                        memcpy(ptr, buffer_.data(), buffer_.size());
                        return true;
                    }
                }
            }

            return false;
        }

        bool listen(const char* route, int* value)
        {
            if (osck_.isOk() && osck_.receiveNextPacket(timeout_))
            {
                oscpkt::PacketReader reader;
                reader.init(osck_.packetData(), osck_.packetSize());
                oscpkt::Message* msg;
                if (reader.isOk() && (msg = reader.popMessage()) != 0)
                {
                    //int num = 0;
                    //msg->arg().popInt32(num);
                    //*value = num;
                    //return true;
                    if (msg->partialMatch(route).isInt32())
                    {
                        msg->partialMatch(route).popInt32(*value);
                        return true;
                    }
                }
            }

            return false;
        }

        void listen(const char* route, float& value)
        {
            listen(route);
            memcpy(&value, buffer_.data(), sizeof(float));
            return;
        }

        bool listen(std::string& value)
        {
            if (osck_.isOk() && osck_.receiveNextPacket(timeout_))
            {
                oscpkt::PacketReader reader;
                reader.init(osck_.packetData(), osck_.packetSize());
                oscpkt::Message* msg;
                if  (reader.isOk() && (msg = reader.popMessage()) != 0)
                {
                    msg->arg().popStr(value);
                    route_.assign(msg->addressPattern());
                    return true;
                }
            } else {
                //std::cout << "[OSC Error] timeout (" << timeout_ << ")" << std::endl;
            }

            return false;
        }

        oscpkt::UdpSocket osck_;
        std::string route_;

    protected:

        size_t size_;
        int timeout_;
        std::vector<char> buffer_;

        oscpkt::PacketWriter* writer_;
};


template <typename T> class OSCListener : OSCServer
{
    public:

        OSCListener(void) : OSCServer()
        {
            size_ = sizeof(T);
        }

        // wait for a new message and return a pointer to it
        T* listen(const char* path)
        {
            while (osck_.isOk()) {
                if (osck_.receiveNextPacket(timeout_))
                {
                    oscpkt::PacketReader reader;
                    reader.init(osck_.packetData(), osck_.packetSize());
                    oscpkt::Message* msg;
                    while (reader.isOk() && (msg = reader.popMessage()) != 0)
                    {
                        if (msg->match(path).popBlob(buffer_).isOkNoMoreArgs())
                        {
                            memcpy(&messages_[0], buffer_.data(), size_);
                        }
                    }
                    break;
                }
            }

            return &messages_[0];
        }

    protected:

        std::vector<T> messages_;

};
