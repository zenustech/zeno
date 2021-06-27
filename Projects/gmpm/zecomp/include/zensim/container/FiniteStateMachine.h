#pragma once
#include "zensim/types/Optional.h"
#include "zensim/types/Polymorphism.h"

namespace zs {

  template <typename Derived, typename StateVariant> class FSM {
  public:
    FSM(StateVariant&& state) : _state(state) {}
    template <typename Event> void dispatch(Event&& event) {
      Derived& self = static_cast<Derived&>(*this);
      auto newState = match(_state)([&](auto& s) -> optional<StateVariant> {
        return self.onEvent(s, std::forward<Event>(event));
      });
      //[&](auto & s) ->optional<StateVariant> { return self.onEvent(s, (event)); });
      if (newState) _state = *std::move(newState);
    }

  private:
    StateVariant _state;
  };

}  // namespace zs
