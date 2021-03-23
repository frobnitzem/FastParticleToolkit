#pragma once

#include <iostream>
#include <chrono>
#include <alpaka/alpaka.hpp>

namespace fpt {

struct Timed {
    const char *name;
    const std::chrono::time_point<std::chrono::steady_clock> beginT;
    std::chrono::time_point<std::chrono::steady_clock> endT;
    Timed(const char *name_) :
        name(name_),
        beginT(std::chrono::steady_clock::now()),
        endT(beginT) {}
    void stop() {
        if(beginT == endT)
            endT = std::chrono::steady_clock::now();
    }
    ~Timed() {
        stop();
        std::cout << "Time for " << name << ": "
                  << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << std::endl;
    }
};

template <typename Q, typename F>
void time_kernel(Q queue, const std::string &name, F fn, int calls=1000) {
    std::cout << name << " (" << calls << " runs)" << std::endl;

    { alpaka::wait(queue);
      fpt::Timed timer(name.c_str());
      for(int i=0; i<calls; i++) {
        fn();
      }
      alpaka::wait(queue);
      //GPUCheckErrors("streams execution error");
    }

    //et1 = dtime_usec(et1);
    //std::cout << "time per call (ms): " << et1/(calls*(float)USECPSEC/1000.0) << std::endl;
}

}
