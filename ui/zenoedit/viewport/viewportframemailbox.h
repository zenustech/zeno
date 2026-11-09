#pragma once
#include <QImage>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <optional>
#include <utility>

// The renderer can outpace Qt, especially while input or painting is busy.
// Keep one replaceable image and at most one queued delivery notification.
// The UI takes the newest image without waiting for the renderer or replaying
// obsolete camera views. Recorded files do not pass through this mailbox.
class ViewportFrameMailbox {
public:
    struct Frame { QImage image; uint64_t sequence{}; QString lodStatus; };
    struct Stats { uint64_t published{},delivered{},replaced{},notifications{}; bool pending{}; };
    bool publish(const QImage& image,const QString& lodStatus={}) {
        std::lock_guard<std::mutex> lock(mutex);
        if(counts.pending)++counts.replaced;
        latest={image,++counts.published,lodStatus};counts.pending=true;
        if(notified)return false;
        notified=true;++counts.notifications;return true;
    }
    std::optional<Frame> take() {
        std::lock_guard<std::mutex> lock(mutex);
        notified=false;
        if(!counts.pending)return std::nullopt;
        counts.pending=false;++counts.delivered;
        return std::exchange(latest,Frame{});
    }
    Stats stats() const { std::lock_guard<std::mutex> lock(mutex);return counts; }
    // Capture the latest actual UI input without adding a queued Qt message
    // for every mouse event while the render thread is busy.
    void noteInput(double time) { newestInput.store(time,std::memory_order_release); }
    double inputTime() const { return newestInput.load(std::memory_order_acquire); }
private:
    mutable std::mutex mutex;
    Frame latest;
    Stats counts;
    bool notified=false;
    std::atomic<double> newestInput{0};
};
