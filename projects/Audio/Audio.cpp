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
            samples.reserve(duration_count);
            for (auto i = 0; i < duration_count; i++) {
                samples.push_back(wave->attr<float>("value")[start_index + i]);
            }
            Aquila::SpectrumType spectrums = fft->fft(samples.data());

            {
                double E = 0;
                for (const auto& spectrum: spectrums) {
                    E += spectrum.real();
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

} // namespace zeno
