#include <iostream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <tuple>
#include <vector>
#include <array>
#include <cassert>
#include <coroutine>

using namespace std;

using std::cout;
using std::endl;
#define show(x) (cout << #x "=" << (x) << endl)

struct Generator {
    struct Promise;

    // compiler looks for promise_type
    using promise_type = Promise;
    coroutine_handle<Promise> coro;

    Generator(coroutine_handle<Promise> h)
        : coro(h) {}

    ~Generator() {
        if (coro)
            coro.destroy();
    }

    // get current value of coroutine
    int value() {
        return coro.promise().val;
    }

    // advance coroutine past suspension
    bool next() {
        coro.resume();
        return !coro.done();
    }

    struct Promise {
        // current value of suspended coroutine
        int val;

        // called by compiler first thing to get coroutine result
        Generator get_return_object() {
            return Generator{coroutine_handle<Promise>::from_promise(*this)};
        }

        // called by compiler first time co_yield occurs
        suspend_always initial_suspend() {
            return {};
        }

        // required for co_yield
        suspend_always yield_value(int x) {
            val=x;
            return {};
        }

        // called by compiler for coroutine without return
        suspend_never return_void() {
            return {};
        }

        // called by compiler last thing to await final result
        // coroutine cannot be resumed after this is called
        suspend_always final_suspend() {
            return {};
        }
    };
};


Generator cpp20_range(int begin, int end) {
    for (int i = begin; i < end; i++) {
        co_yield i;
    }
}


int main(void)
{
    /*auto r = range(2, 4);
    for (auto it = r.begin(); it; ++it) {
        show(*it);
    }*/
    auto res = cpp20_range(2, 4);
    for (int i = 2; i < 4; i++) {
        res.next();
        show(res.value());
    }
}
