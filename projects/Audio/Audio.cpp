#include <zeno/utils/nowarn.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/StringObject.h>
#include <zeno/utils/string.h>
#include <zeno/utils/vec.h>
#include <zeno/zeno.h>
#include "zeno/utils/logger.h"
#include "zeno/types/UserData.h"
#include "zeno/types/StringObject.h"
#include "zeno/types/NumericObject.h"
#include "aquila/aquila/aquila.h"
#include <deque>
#include <zeno/types/ListObject.h>
#include "AudioFile.h"

#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT
#include "minimp3.h"

#include <algorithm>

int calcFrameCountByAudio(std::string path, int fps) {
    AudioFile<float> wav;
    wav.load (path);
    uint64_t ret = wav.getNumSamplesPerChannel();
    ret = ret * fps / wav.getSampleRate();
    return ret + 1;
}

namespace zeno {
    struct ReadWavFile : zeno::INode {
        virtual void apply() override {
            auto path = get_input<StringObject>("path")->get(); // std::string
            AudioFile<float> wav;
            wav.load (path);
            wav.printSummary();

            auto result = std::make_shared<PrimitiveObject>(); // std::shared_ptr<PrimitiveObject>
            result->resize(wav.getNumSamplesPerChannel());
            auto &value = result->add_attr<float>("value"); //std::vector<float>
            auto &t = result->add_attr<float>("t");

            for (std::size_t i = 0; i < result->verts.size(); ++i) {
                value[i] = wav.samples[0][i];
                t[i] = float(i);
            }

            result->userData().set("SampleRate", std::make_shared<zeno::NumericObject>((int)wav.getSampleRate()));
            result->userData().set("BitDepth", std::make_shared<zeno::NumericObject>((int)wav.getBitDepth()));
            result->userData().set("NumSamplesPerChannel", std::make_shared<zeno::NumericObject>((int)wav.getNumSamplesPerChannel()));
            result->userData().set("LengthInSeconds", std::make_shared<zeno::NumericObject>((float)wav.getLengthInSeconds()));

            set_output("wave",result);
        }
    };
    ZENDEFNODE(ReadWavFile, {
        {
            {"readpath", "path"},
        },
        {
            "wave",
        },
        {},
        {
            "audio"
        },
    });

    struct ReadMp3File : zeno::INode {
        virtual void apply() override {
            auto path = get_input<StringObject>("path")->get(); // std::string

            // open the file:
            std::ifstream file(path, std::ios::binary);
            // read the data:
            auto data = std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

            static mp3dec_t mp3d;
            mp3dec_init(&mp3d);

            mp3dec_frame_info_t info;
            float pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];
            int mp3len = 0;
            int sample_len = 0;

            std::vector<float> decoded_data;
            decoded_data.reserve(44100 * 30);
            while (true) {
                int samples = mp3dec_decode_frame(&mp3d, data.data() + mp3len, data.size() - mp3len, pcm, &info);
                if (samples == 0) {
                    break;
                }
                sample_len += samples;
                mp3len += info.frame_bytes;
                for (auto i = 0; i < samples * info.channels; i += info.channels) {
                    decoded_data.push_back(pcm[i]);
                }
            }
            auto result = std::make_shared<PrimitiveObject>(); // std::shared_ptr<PrimitiveObject>
            result->resize(sample_len);
            auto &value = result->add_attr<float>("value"); //std::vector<float>
            auto &t = result->add_attr<float>("t");
            for (std::size_t i = 0; i < result->verts.size(); ++i) {
                value[i] = decoded_data[i];
                t[i] = float(i);
            }
            result->userData().set("SampleRate",std::make_shared<zeno::NumericObject>((int)info.hz));
            result->userData().set("NumSamplesPerChannel", std::make_shared<zeno::NumericObject>((int)sample_len));

            set_output("wave", result);
        }
    };
    ZENDEFNODE(ReadMp3File, {
        {
            {"readpath", "path"},
        },
        {
            "wave",
        },
        {},
        {
            "audio"
        },
    });


    struct AudioBeats : zeno::INode {
        std::deque<double> H;
        virtual void apply() override {
            auto wave = get_input<PrimitiveObject>("wave");
            float threshold = get_input<NumericObject>("threshold")->get<float>();
            auto start_time = get_input<NumericObject>("time")->get<float>();
            float sampleFrequency = wave->userData().get<zeno::NumericObject>("SampleRate")->get<float>();
            int start_index = int(sampleFrequency * start_time);
            int duration_count = 1024;
            auto fft = Aquila::FftFactory::getFft(duration_count);
            std::vector<double> samples;
            samples.resize(duration_count);
            for (auto i = 0; i < duration_count; i++) {
//                if (start_index + i >= wave->size()) {
//                    break;
//                }
                samples[i] = wave->attr<float>("value")[min((start_index + i), wave->size()-1)];
                
                //if (start_index + i >= wave->size()) {
                //    break;
                //}
                //samples[i] = wave->attr<float>("value")[start_index + i];
            }
            Aquila::SpectrumType spectrums = fft->fft(samples.data());

            {
                double E = 0;
                for (const auto& spectrum: spectrums) {
                    E += spectrum.real() * spectrum.real() + spectrum.imag() * spectrum.imag();
                }
                E /= duration_count;
                H.push_back(E);
            }

            while (H.size() > 43) {
                H.pop_front();
            }
            double avg_H = 0;
            for (const auto& E: H) {
                avg_H += E;
            }
            avg_H /= H.size();

            double var_H = 0;
            for (const auto& E: H) {
                var_H += (E - avg_H) * (E - avg_H);
            }
            var_H /= H.size();
            int beat = H.back() - threshold > (-15 * var_H + 1.55) * avg_H;
            set_output("beat", std::make_shared<NumericObject>(beat));
            set_output("var_H", std::make_shared<NumericObject>((float)var_H));


            auto output_H = std::make_shared<ListObject>();
            for (int i = 0; i < 43 - H.size(); i++) {
                output_H->arr.emplace_back(std::make_shared<NumericObject>((float)0));
            }
            for (const auto & h: H) {
                output_H->arr.emplace_back(std::make_shared<NumericObject>((float)h));
            }
            set_output("H", output_H);

            auto output_E = std::make_shared<ListObject>();
            for (const auto& spectrum: spectrums) {
                double e = spectrum.real() * spectrum.real() + spectrum.imag() * spectrum.imag();
                output_E->arr.emplace_back(std::make_shared<NumericObject>((float)e));
            }
            set_output("E", output_E);
        }
    };

ZENDEFNODE(AudioBeats, {
        {
            "wave",
            {"float", "time", "0"},
            {"float", "threshold", "0.005"},
        },
        {
            "beat",
            "var_H",
            "H",
            "E",
        },
        {},
        {
            "audio"
        },
    });

struct AudioEnergy : zeno::INode {
    double minE = std::numeric_limits<double>::max();
    double maxE = std::numeric_limits<double>::min();
    std::vector<double> init;
    virtual void apply() override {
        auto wave = get_input<PrimitiveObject>("wave");
        int duration_count = 1024;
        if (init.empty()) {
            auto fft = Aquila::FftFactory::getFft(duration_count);
            int clip_count = wave->size() / duration_count;
            init.reserve(clip_count);
            for (auto i = 0; i < clip_count; i++) {
                std::vector<double> samples;
                samples.resize(duration_count);
                for (auto j = 0; j < duration_count; j++) {
                    samples[j] = wave->attr<float>("value")[min(duration_count * i + j, wave->size()-1)];
                }
                Aquila::SpectrumType spectrums = fft->fft(samples.data());
                {
                    double E = 0;
                    for (const auto& spectrum: spectrums) {
                        E += spectrum.real() * spectrum.real() + spectrum.imag() * spectrum.imag();
                    }
                    E /= duration_count;
                    minE = min(minE, E);
                    maxE = max(maxE, E);
                    init.push_back(E);
                }
            }
//            for (auto i = 0; i < clip_count; i++) {
//                init[i] = init[i] / maxE;
//            }
        }

//        auto vis = std::make_shared<PrimitiveObject>();
//        vis->resize(init.size());
//        auto &index = vis->add_attr<float>("index");
//        auto &listE = vis->add_attr<float>("E");
//        for (auto i = 0; i < init.size(); i++) {
//            index[i] = i;
//            listE[i] = init[i];
//        }
//        set_output("vis", vis);

        set_output("minE", std::make_shared<NumericObject>((float)minE));
        set_output("maxE", std::make_shared<NumericObject>((float)maxE));

        auto start_time = get_input2<float>("time");
        float sampleFrequency = wave->userData().get<zeno::NumericObject>("SampleRate")->get<float>();
        int start_index = int(sampleFrequency * start_time);
        auto fft = Aquila::FftFactory::getFft(duration_count);
        std::vector<double> samples;
        samples.resize(duration_count);
        for (auto i = 0; i < duration_count; i++) {
            samples[i] = wave->attr<float>("value")[min((start_index + i), wave->size()-1)];
        }
        Aquila::SpectrumType spectrums = fft->fft(samples.data());
        double E = 0;
        for (const auto& spectrum: spectrums) {
            E += spectrum.real() * spectrum.real() + spectrum.imag() * spectrum.imag();
        }
        E /= duration_count;
        set_output("E", std::make_shared<NumericObject>((float)E));
        double uniE = (E - minE) / (maxE - minE);
        set_output("uniE", std::make_shared<NumericObject>((float)E));
    }
};
    ZENDEFNODE(AudioEnergy, {
        {
            "wave",
            {"float", "time", "0"},
        },
        {
            "E",
            "uniE",
            "minE",
            "maxE",
//            "vis",
        },
        {},
        {
            "audio"
        },
    });

} // namespace zeno
