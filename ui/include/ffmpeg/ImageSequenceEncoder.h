#pragma once
#include <zeno/utils/log.h>
// ===== FFmpeg =====
// FFmpeg是一个强大的多媒体处理库，用于处理视频、音频等多媒体数据
// extern "C" 确保C语言的FFmpeg库在C++中正确链接
// 因为FFmpeg是用C语言编写的，而我们的代码是C++
extern "C" {
#include <libavformat/avformat.h>  // 处理视频容器格式（如MP4、AVI等）
#include <libavcodec/avcodec.h>   // 视频编解码器相关功能
#include <libswscale/swscale.h>   // 图像缩放和格式转换
}

// ===== stb_image =====
// stb_image是一个轻量级的图像加载库，用于读取各种格式的图片文件
#include <string>

//#ifndef STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_STATIC
//#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"  // 包含stb_image库的头文件
//#endif

/**
 * ImageSequenceEncoder类：将一系列图片合成为视频文件
 * 
 * 这个类的主要功能是：
 * 1. 读取多张图片（如PNG、JPG等格式）
 * 2. 将这些图片按顺序合成为一个视频文件（如MP4格式）
 * 3. 支持设置帧率、码率等视频参数
 * 
 * 使用流程：
 * 1. 调用open()方法设置输出路径和参数
 * 2. 多次调用addFrameFromFile()方法添加图片
 * 3. 调用close()方法完成视频合成
 */
class ImageSequenceEncoder {
public:
    // 默认构造函数：使用编译器生成的默认实现
    ImageSequenceEncoder() = default;
    ~ImageSequenceEncoder() { close(); }

    /**
     * 打开视频编码器（延迟初始化）
     * 
     * 这个方法只是设置参数，真正的FFmpeg初始化会在第一帧图片到达时进行
     * 这样设计的好处是：可以根据第一张图片的尺寸动态确定视频分辨率
     * 
     * @param outPath 输出视频文件路径（如："output.mp4"）
     * @param fps 视频帧率（每秒多少帧，如30表示30帧/秒）
     * @param bitrateK 视频码率（千比特/秒，如8000表示8Mbps）
     * @return 是否成功打开（这里总是返回true，真正的初始化在后面）
     */
    bool open(const std::string& outPath,
        int fps,
        int bitrateK)
    {
        resetState();    // 重置所有状态，准备新的编码会话
        ok = true;       // 标记操作状态为成功

        // 保存用户设置的参数
        outputPath = outPath;  // 输出文件路径
        this->fps = fps;       // 帧率
        this->bitrateK = bitrateK; // 码率（千比特/秒）
        
        opened = true;          // 标记为"逻辑打开"状态（参数已设置）
        initialized = false;    // 标记为"尚未初始化"（等待第一帧图片）
        return true;
    }

    /**
     * 添加一帧图片到视频中
     * 
     * 这个方法执行以下操作：
     * 1. 如果是第一帧：初始化FFmpeg编码器（根据图片尺寸）
     * 2. 加载图片数据到内存
     * 3. 将RGB格式转换为YUV格式（视频编码标准格式）
     * 4. 设置时间戳（决定帧在视频中出现的时间）
     * 5. 编码并写入视频文件
     * 
     * @param filePath 图片文件路径
     * @return 是否成功添加帧
     */
    bool addFrameFromFile(const std::string& filePath)
    {
        // 检查状态：必须先调用open()且之前没有错误
        if (!opened || !ok)
            return false;

        // 使用stb_image库加载图片
        // stbi_load参数说明：
        // - filePath.c_str(): 图片文件路径
        // - &w, &h: 获取图片的宽度和高度
        // - &comp: 获取原始通道数（如RGB=3，RGBA=4）
        // - 3: 强制转换为RGB三通道（忽略透明度等）
        stbi_set_flip_vertically_on_load(false);
        int w, h, comp;
        unsigned char* data = stbi_load(filePath.c_str(), &w, &h, &comp, 3);
        stbi_set_flip_vertically_on_load(true);

        // 检查图片是否加载成功
        if (!data) {
            ok = false;  // 标记操作失败
            return false;
        }

        // 延迟初始化：如果是第一帧图片，根据其尺寸初始化编码器
        if (!initialized) {
            // 初始化FFmpeg编码器（使用第一帧的尺寸）
            if (!initWithFirstFrame(w, h)) {
                stbi_image_free(data);  // 释放图片内存
                ok = false;
                return false;
            }
            initialized = true;  // 标记为已初始化
        }
        else {
            // 后续帧：检查尺寸是否与第一帧一致
            // 视频编码要求所有帧尺寸相同
            if (w != width || h != height) {
                stbi_image_free(data);
                ok = false;
                return false;
            }
        }

        // 准备RGB源数据用于格式转换
        // src: 指向RGB数据的指针数组（只有一个平面）
        // srcStride: 每行的字节数（宽度×3字节，因为RGB每个像素3字节）
        uint8_t* src[] = { data };
        int srcStride[] = { w * 3 };

        // 使用FFmpeg的sws_scale函数进行格式转换
        // 将RGB格式转换为YUV420P格式（视频编码标准格式）
        sws_scale(
            sws,           // 格式转换器上下文
            src, srcStride, // 源数据和步长
            0, height,     // 转换范围（从第0行到height行）
            frame->data, frame->linesize // 目标数据和步长
        );

        // 设置时间戳（Presentation Time Stamp）
        // PTS决定帧在视频中出现的时间点
        // 这里使用简单的帧索引递增：第0帧、第1帧、第2帧...
        frame->pts = frameIndex++;

        // 将帧发送给编码器进行编码
        // avcodec_send_frame: 将一帧数据交给编码器
        if (avcodec_send_frame(codec, frame) < 0) {
            stbi_image_free(data);
            ok = false;
            return false;
        }

        // 使用FFmpeg新的AVPacket API（更安全的内存管理）
        // AVPacket: 编码后的数据包容器
        AVPacket* pkt = av_packet_alloc();
        if (!pkt) {
            stbi_image_free(data);
            ok = false;
            return false;
        }

        // 循环接收编码器输出的数据包
        // 一帧输入可能产生多个数据包（特别是关键帧）
        while (avcodec_receive_packet(codec, pkt) == 0) {
            // ★ 关键步骤：时间基转换
            // 将编码器时间基转换为流时间基
            // 时间基：表示时间单位，如{1,30}表示1/30秒
            av_packet_rescale_ts(
                pkt,                    // 要转换的数据包
                codec->time_base,       // 源时间基（编码器的时间单位）
                stream->time_base       // 目标时间基（流的时间单位）
            );
            pkt->stream_index = stream->index;  // 设置流索引

            // 将数据包写入视频文件
            // av_interleaved_write_frame: 交错写入（保证音视频同步）
            if (av_interleaved_write_frame(fmt, pkt) < 0) {
                // 写入失败：清理资源并返回错误
                av_packet_unref(pkt);   // 释放数据包内容
                av_packet_free(&pkt);   // 释放数据包本身
                stbi_image_free(data);
                ok = false;
                return false;
            }
            av_packet_unref(pkt);  // 释放当前数据包内容，准备接收下一个
        }

        // 释放数据包和图片内存
        av_packet_free(&pkt);
        stbi_image_free(data);
        return true;
    }

    /**
     * 完成视频合成并关闭编码器
     * 
     * 这个方法执行以下关键操作：
     * 1. 发送空帧刷新编码器缓冲区（获取所有剩余数据包）
     * 2. 写入视频文件尾部信息
     * 3. 释放所有FFmpeg资源
     * 
     * 注意：必须调用此方法才能生成完整的视频文件
     * 
     * @return 视频合成是否成功（只有所有帧都成功编码才返回true）
     */
    bool close()
    {
        // 检查是否已经调用过open()
        if (!opened)
            return false;

        // 只有在成功初始化且没有错误的情况下才进行最终处理
        if (ok && initialized) {
            // ★ 关键步骤：发送空帧刷新编码器
            // 编码器内部有缓冲区，发送nullptr会强制编码器输出所有缓冲数据
            avcodec_send_frame(codec, nullptr);

            // 使用新的AVPacket API处理剩余的编码数据
            AVPacket* pkt = av_packet_alloc();
            if (pkt) {
                // 循环接收编码器最后输出的数据包
                while (avcodec_receive_packet(codec, pkt) == 0) {
                    // 时间基转换（与addFrameFromFile中相同）
                    av_packet_rescale_ts(
                        pkt,
                        codec->time_base,
                        stream->time_base
                    );
                    pkt->stream_index = stream->index;

                    // 写入剩余的数据包
                    av_interleaved_write_frame(fmt, pkt);
                    av_packet_unref(pkt);  // 释放当前数据包内容
                }
                av_packet_free(&pkt);  // 释放数据包容器
            }

            // ★ 关键步骤：写入视频文件尾部
            // 这会在MP4文件中写入必要的结束标记和元数据
            av_write_trailer(fmt);
        }

        // 清理所有FFmpeg资源
        cleanup();
        
        // 返回最终结果：只有成功初始化且没有错误才算成功
        return ok && initialized;
    }

private:
    /**
     * 根据第一帧图片的尺寸初始化FFmpeg编码器
     * 
     * 这是整个编码过程的核心初始化方法，创建所有必要的FFmpeg对象：
     * 1. 输出格式上下文（容器）
     * 2. 编码器上下文
     * 3. 视频流
     * 4. 帧缓冲区和格式转换器
     * 
     * @param w 图片宽度（像素）
     * @param h 图片高度（像素）
     * @return 初始化是否成功
     */
    bool initWithFirstFrame(int w, int h)
    {
        width = w;
        height = h;

        // 创建输出容器（MP4）
        if (avformat_alloc_output_context2(
            &fmt, nullptr, "mp4", outputPath.c_str()) < 0)
            return false;

        const AVCodec* enc = nullptr;

        // =========================
        // 1. 先尝试 H264
        // =========================
        {
            const AVCodec* h264 = avcodec_find_encoder(AV_CODEC_ID_H264);
            if (h264) {
                AVCodecContext* tmp = avcodec_alloc_context3(h264);
                if (tmp) {
                    tmp->codec_type = AVMEDIA_TYPE_VIDEO;
                    tmp->width = width;
                    tmp->height = height;
                    tmp->pix_fmt = AV_PIX_FMT_YUV420P;
                    tmp->time_base = { 1, fps };
                    tmp->framerate = { fps, 1 };
                    tmp->bit_rate = bitrateK * 1000;
                    tmp->gop_size = 12;
                    tmp->max_b_frames = 0;

                    if (fmt->oformat->flags & AVFMT_GLOBALHEADER)
                        tmp->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

                    AVDictionary* opts = nullptr;
                    av_dict_set(&opts, "preset", "fast", 0);
                    av_dict_set(&opts, "tune", "zerolatency", 0);

                    if (avcodec_open2(tmp, h264, &opts) == 0) {
                        // ✅ H264 真正可用
                        enc = h264;
                        codec = tmp;
                        av_dict_free(&opts);
                    }
                    else {
                        // ❌ H264 不可用，彻底释放
                        av_dict_free(&opts);
                        avcodec_free_context(&tmp);
                    }
                }
            }
        }

        // =========================
        // 2. H264 不可用 → 回退 MPEG4
        // =========================
        if (!enc) {
            enc = avcodec_find_encoder(AV_CODEC_ID_MPEG4);
            if (!enc)
                return false;

            codec = avcodec_alloc_context3(enc);
            if (!codec)
                return false;

            codec->codec_type = AVMEDIA_TYPE_VIDEO;
            codec->width = width;
            codec->height = height;
            codec->pix_fmt = AV_PIX_FMT_YUV420P;
            codec->time_base = { 1, fps };
            codec->framerate = { fps, 1 };
            codec->bit_rate = bitrateK * 1000;
            codec->gop_size = 12;
            codec->max_b_frames = 0;

            if (fmt->oformat->flags & AVFMT_GLOBALHEADER)
                codec->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

            if (avcodec_open2(codec, enc, nullptr) < 0)
                return false;
        }

        // =========================
        // 3. 创建视频流
        // =========================
        stream = avformat_new_stream(fmt, enc);
        if (!stream)
            return false;

        stream->time_base = codec->time_base;

        if (avcodec_parameters_from_context(stream->codecpar, codec) < 0)
            return false;

        // =========================
        // 4. 打开输出文件
        // =========================
        if (avio_open(&fmt->pb, outputPath.c_str(), AVIO_FLAG_WRITE) < 0)
            return false;

        av_dump_format(fmt, 0, outputPath.c_str(), 1);

        if (avformat_write_header(fmt, nullptr) < 0)
            return false;

        // =========================
        // 5. 分配帧与转换器
        // =========================
        frame = av_frame_alloc();
        if (!frame)
            return false;

        frame->format = codec->pix_fmt;
        frame->width = width;
        frame->height = height;

        if (av_frame_get_buffer(frame, 0) < 0)
            return false;

        sws = sws_getContext(
            width, height, AV_PIX_FMT_RGB24,
            width, height, AV_PIX_FMT_YUV420P,
            SWS_BILINEAR, nullptr, nullptr, nullptr);

        return sws != nullptr;
    }

    /**
     * 重置所有状态，准备新的编码会话
     * 
     * 这个方法在open()开始时调用，确保从一个干净的状态开始
     */
    void resetState()
    {
        cleanup();           // 清理之前的资源
        ok = true;           // 重置操作状态为成功
        opened = false;      // 标记为未打开
        initialized = false; // 标记为未初始化
        frameIndex = 0;      // 重置帧计数器
    }

    /**
     * 清理所有FFmpeg资源
     * 
     * FFmpeg资源释放规则：
     * 1. 先释放内部资源（如转换器、帧、编码器）
     * 2. 再释放容器资源（如格式上下文）
     * 3. 所有指针置空防止悬空指针
     */
    void cleanup()
    {
        // 释放格式转换器
        if (sws)   sws_freeContext(sws);
        
        // 释放帧缓冲区
        if (frame) av_frame_free(&frame);
        
        // 释放编码器上下文
        if (codec) avcodec_free_context(&codec);

        // 释放格式上下文（视频文件容器）
        if (fmt) {
            // 先关闭文件IO
            if (fmt->pb)
                avio_closep(&fmt->pb);
            // 再释放格式上下文本身
            avformat_free_context(fmt);
        }

        // 所有指针置空（重要：防止悬空指针）
        sws = nullptr;
        frame = nullptr;
        codec = nullptr;
        fmt = nullptr;
        stream = nullptr;
        
        // 重置状态标志
        opened = false;
        initialized = false;
    }

private:
    // ===== FFmpeg核心对象 =====
    AVFormatContext* fmt = nullptr;    // 格式上下文：管理视频文件容器（如MP4）
    AVCodecContext* codec = nullptr;  // 编码器上下文：管理视频编码参数和状态
    AVStream* stream = nullptr;        // 视频流：容器中的单个视频流
    SwsContext* sws = nullptr;         // 格式转换器：RGB到YUV格式转换
    AVFrame* frame = nullptr;           // 帧缓冲区：存储转换后的YUV数据

    // ===== 编码参数和状态 =====
    std::string outputPath;            // 输出视频文件路径
    int  fps = 0;                       // 帧率（帧/秒）
    int  bitrateK = 0;                  // 码率（千比特/秒）
    int  width = 0;                     // 视频宽度（像素）
    int  height = 0;                    // 视频高度（像素）
    int  frameIndex = 0;                // 当前帧索引（用于时间戳计算）

    // ===== 状态标志 =====
    bool opened = false;                // open()方法是否被调用（参数已设置）
    bool initialized = false;           // 是否已用第一帧初始化FFmpeg
    bool ok = true;                     // 操作是否成功（错误时设为false）
};

static bool videoCompose(QString imgPath, QString outPath, int startFrame, int fps, int bitrate) {
    // 计算总帧数
    int totalFrames = 0;
    int currentFrame = startFrame;
    while (true) {
        QString currentImgPath = QString::fromStdString(zeno::format(imgPath.toStdString(), currentFrame));
        if (!QFile::exists(currentImgPath)) {
            break;
        }
        totalFrames++;
        currentFrame++;
    }

    // 开始合成
    ImageSequenceEncoder enc;
    bool ret = enc.open(
        outPath.toStdString(),
        fps,
        bitrate
    );

    if (!ret) {
        zeno::log_info("Failed to open output file.");
        return false;
    }

    currentFrame = startFrame;
    int processedFrames = 0;
    bool composeSuccess = true;
    while (true) {
        QString currentImgPath = QString::fromStdString(zeno::format(imgPath.toStdString(), currentFrame));
        if (!QFile::exists(currentImgPath)) {
            break;
        }

        ret = enc.addFrameFromFile(currentImgPath.toStdString());
        if (!ret) {
            composeSuccess = false;
            break;
        }

        processedFrames++;
        currentFrame++;
    }

    // 即使close返回false，只要成功处理了所有帧，就认为是成功的
    enc.close();

    // 如果成功处理了至少一帧，并且没有中途失败，就认为是成功的
    bool finalSuccess = composeSuccess && processedFrames > 0;
    return finalSuccess;
}