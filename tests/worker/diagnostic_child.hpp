#pragma once
// Extracted from cpu_group_numeric_diagnostic: only unreaped children we fork may be signalled.
#include "strata/worker/expert_client.hpp"
#include <algorithm>
#include <cerrno>
#include <csignal>
#include <chrono>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <fcntl.h>
#include <poll.h>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <set>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#ifdef __linux__
#include <sched.h>
#include <sys/prctl.h>
#endif
namespace strata::worker::diagnostic {
inline void need(bool ok,const std::string& why){if(!ok)throw std::runtime_error(why);}
inline uint64_t millis(){return std::chrono::duration_cast<std::chrono::milliseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();}
struct ChildSpec {
    std::string binary,root,manifest,log_path;
    uint64_t budget_mib=256;
    int workers=12,loaders=4,frame_ms=10000,setup_ms=30000,start_ms=10000;
    std::vector<int> setup_cpus;
};
#ifdef __linux__
inline cpu_set_t mask_of(const std::vector<int>& cpus){
    cpu_set_t mask;CPU_ZERO(&mask);
    for(int cpu:cpus){need(cpu>=0 && cpu<CPU_SETSIZE,"CPU index outside affinity bounds");CPU_SET(cpu,&mask);}
    need(CPU_COUNT(&mask)>0,"empty CPU mask");return mask;
}
#endif
struct Child {
    pid_t pid=-1;int pipe=-1,log_fd=-1;uint16_t port=0;
    std::string log;bool exit_ok=false;int exit_status=0;
    Child(const Child&)=delete;Child& operator=(const Child&)=delete;
    bool owns_listener() const {
#ifdef __linux__
        if(pid<=0)return false;
        std::set<std::string> sockets;std::ifstream net("/proc/net/tcp");std::string line;
        char address[32];std::snprintf(address,sizeof address,"0100007F:%04X",(unsigned)port);
        while(std::getline(net,line)){std::istringstream row(line);std::vector<std::string> fields;std::string value;
            while(row>>value)fields.push_back(value);
            if(fields.size()>9 && fields[1]==address && fields[3]=="0A")sockets.insert("socket:["+fields[9]+"]");}
        for(const auto& fd:std::filesystem::directory_iterator("/proc/"+std::to_string(pid)+"/fd")){
            std::error_code error;const auto target=std::filesystem::read_symlink(fd.path(),error);
            if(!error && sockets.count(target.string()))return true;}
        return false;
#else
        return pid>0; // existing non-Linux fixture; configured local-subset mode is Linux-only
#endif
    }
    void pump(){
        char data[4096];
        for(;;){const auto n=::read(pipe,data,sizeof data);
            if(n>0){need(log.size()+(size_t)n<=1u<<20,"owned worker output exceeds 1 MiB");log.append(data,n);
                size_t at=0;while(log_fd>=0 && at<(size_t)n){const auto wrote=::write(log_fd,data+at,n-at);
                    if(wrote<0 && errno==EINTR)continue;need(wrote>0,"owned worker log write failed");at+=(size_t)wrote;}}
            else if(n<0 && errno==EINTR)continue;else break;
        }
    }
    bool wait(int seconds,int& status){
        if(pid<=0)return false;
        const auto end=millis()+seconds*1000;
        do{pump();const auto got=::waitpid(pid,&status,WNOHANG);
            if(got==pid){pid=-1;exit_status=status;exit_ok=WIFEXITED(status)&&WEXITSTATUS(status)==0;return true;}
            need(got>=0 || errno==EINTR,"owned worker waitpid failed");
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }while(millis()<end);return false;
    }
    void abort() noexcept {
        if(pid>0){::kill(pid,SIGTERM);int status=0;
            try{if(wait(5,status))return;}catch(...){}
            ::kill(pid,SIGKILL);while(::waitpid(pid,&status,0)<0 && errno==EINTR){}
            pid=-1;exit_status=status;exit_ok=false;
        }
    }
    explicit Child(const ChildSpec& s){
        need(std::filesystem::path(s.binary).is_absolute() && ::access(s.binary.c_str(),X_OK)==0,
             "worker executable must be an existing absolute path");
        std::vector<std::string> args{s.binary,"--port","0","--workers",std::to_string(s.workers),
            "--local-load-threads",std::to_string(s.loaders),"--memory-budget-mib",std::to_string(s.budget_mib),
            "--local-root",s.root,"--local-manifest",s.manifest,"--deadline-ms",std::to_string(s.frame_ms),
            "--local-setup-ms",std::to_string(s.setup_ms),"--residency-forever"};
        std::vector<char*> argv;for(auto& a:args)argv.push_back(a.data());argv.push_back(nullptr);
#ifdef __linux__
        cpu_set_t setup{};if(!s.setup_cpus.empty())setup=mask_of(s.setup_cpus);
#else
        need(s.setup_cpus.empty(),"configured owned affinity requires Linux");
#endif
        int fds[2];need(::pipe(fds)==0,"owned worker pipe");
        if(::fcntl(fds[0],F_SETFD,FD_CLOEXEC) || ::fcntl(fds[1],F_SETFD,FD_CLOEXEC)){
            ::close(fds[0]);::close(fds[1]);throw std::runtime_error("owned worker close-on-exec");}
        if(!s.log_path.empty())log_fd=::open(s.log_path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600);
        if(!s.log_path.empty() && log_fd<0){::close(fds[0]);::close(fds[1]);throw std::runtime_error("exclusive owned worker log");}
        const pid_t parent=::getpid();pid=::fork();
        if(pid<0){::close(fds[0]);::close(fds[1]);if(log_fd>=0)::close(log_fd);throw std::runtime_error("owned worker fork");}
        if(pid==0){
#ifdef __linux__
            if(::prctl(PR_SET_PDEATHSIG,SIGTERM) || ::getppid()!=parent)_exit(126);
            if(!s.setup_cpus.empty() && ::sched_setaffinity(0,sizeof setup,&setup))_exit(126);
#endif
            ::close(fds[0]);if(::dup2(fds[1],STDOUT_FILENO)<0 || ::dup2(fds[1],STDERR_FILENO)<0)_exit(126);
            ::close(fds[1]);::execv(argv[0],argv.data());_exit(127);
        }
        ::close(fds[1]);pipe=fds[0];
        try{need(::fcntl(pipe,F_SETFL,O_NONBLOCK)==0,"owned worker nonblocking pipe");
            const auto end=millis()+s.start_ms;
            while(!port && millis()<end){pump();const auto at=log.find("listening on 127.0.0.1:");
                if(at!=std::string::npos && log.find('\n',at)!=std::string::npos){
                    const auto value=std::stoul(log.substr(at+23));need(value>0 && value<=65535,"owned worker port");port=(uint16_t)value;}
                if(!port){int status=0;const auto done=::waitpid(pid,&status,WNOHANG);
                    if(done==pid){pid=-1;exit_status=status;throw std::runtime_error("owned worker exited before listening");}
                    need(done==0 || (done<0 && errno==EINTR),"owned worker startup waitpid");
                    pollfd p{pipe,POLLIN,0};::poll(&p,1,20);}}
            need(port && log.find(", "+std::to_string(s.workers)+" pool workers")!=std::string::npos,"owned worker startup/count");
            need(owns_listener(),"listening socket is not owned by the retained child");
        }catch(...){abort();::close(pipe);pipe=-1;if(log_fd>=0)::close(log_fd);log_fd=-1;throw;}
    }
    ~Child(){abort();if(pipe>=0)::close(pipe);if(log_fd>=0)::close(log_fd);}
    void unload(){ExpertWorkerClient c;std::string e;
        need(owns_listener(),"refuse unload without owned listener");
        need(c.connect_host("127.0.0.1",port,5000,e),e);need(c.unload(e),e);}
    void stop(){
        need(pid>0 && owns_listener(),"refuse stop without owned child/listener");ExpertWorkerClient c;std::string e;
        need(c.connect_host("127.0.0.1",port,5000,e),e);
        const bool acknowledged=c.shutdown(e);c.close();int status=0;
        // A lost STOP acknowledgement is uncertain. Never resend; reconcile via our retained child.
        const bool reaped=wait(10,status);
        need(reaped && exit_ok,acknowledged?"owned worker clean shutdown/reap failed":"owned STOP uncertain; child not confirmed clean");
    }
    void compute_affinity(int caller,const std::vector<int>& pool){
#ifdef __linux__
        need(pid>0,"owned affinity after reap");
        std::vector<int> tids;
        for(const auto& entry:std::filesystem::directory_iterator("/proc/"+std::to_string(pid)+"/task"))
            tids.push_back(std::stoi(entry.path().filename()));
        std::sort(tids.begin(),tids.end());
        need(tids.size()==pool.size()+2 && tids.front()==pid,"owned READY task count mismatch");
        for(size_t i=0;i<=pool.size();++i){auto mask=mask_of({i?pool[i-1]:caller});
            need(::sched_setaffinity(tids[i],sizeof mask,&mask)==0,"owned compute affinity apply failed");
            cpu_set_t actual;CPU_ZERO(&actual);
            need(::sched_getaffinity(tids[i],sizeof actual,&actual)==0 && CPU_EQUAL(&actual,&mask),"owned compute affinity verify failed");}
#else
        (void)caller;(void)pool;throw std::runtime_error("owned compute affinity requires Linux");
#endif
    }
};
}
