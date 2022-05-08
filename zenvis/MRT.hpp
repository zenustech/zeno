// MRT = Multiple Render targets
#ifndef __MRT_H__
#define __MRT_H__

#include "stdafx.hpp"
#include <iostream>
#include <vector>

namespace zenvis
{
    struct MRT
    {
    public:
        bool isUse{false};
        bool isInit{false};

        GLuint msFbo{GL_NONE};
        GLuint fbo{GL_NONE};
        GLuint depthRbo{GL_NONE};
        GLuint colorRbo{GL_NONE};
        GLuint colorTex{GL_NONE};
        GLuint positionRbo{GL_NONE};
        GLuint positionTex{GL_NONE};
        GLuint normalRbo{GL_NONE};
        GLuint normalTex{GL_NONE};
        GLuint texcoordRbo{GL_NONE};
        GLuint texcoordTex{GL_NONE};
        GLuint tangentRbo{GL_NONE};
        GLuint tangentTex{GL_NONE};
        static constexpr unsigned int attachmentCount{5};

    private:
        MRT() = default;
        ~MRT() = default;
    
    public:
        MRT(const MRT &) = delete;
        MRT(MRT &&) = delete;
        MRT &operator=(const MRT &) = delete;
        MRT &operator=(MRT &) = delete;
        static MRT &getInstance()
        {
            static MRT mrt;
            return mrt;
        }

        void init(GLsizei samples, GLsizei width, GLsizei height)
        {
            if (isInit)
            {
                return;
            }

            CHECK_GL(glGenFramebuffers(1, &msFbo));
            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, msFbo));

            CHECK_GL(glGenRenderbuffers(1, &depthRbo));
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, depthRbo));
            CHECK_GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT32F, width, height));
            CHECK_GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRbo));

            unsigned int counter = 0;
            auto initRbo = [samples, width, height, &counter](GLuint &rbo)
            {
                CHECK_GL(glGenRenderbuffers(1, &rbo));
                CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbo));
                CHECK_GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA32F, width, height));
                CHECK_GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + counter, GL_RENDERBUFFER, rbo));
                ++counter;
            };

            initRbo(colorRbo);
            initRbo(positionRbo);
            initRbo(normalRbo);
            initRbo(texcoordRbo);
            initRbo(tangentRbo);

            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, GL_NONE));

            std::vector<GLenum> attachments;
            attachments.reserve(attachmentCount);
            for (unsigned int i = 0; i < attachmentCount; ++i)
            {
                attachments.push_back(GL_COLOR_ATTACHMENT0 + i);
            }
            CHECK_GL(glDrawBuffers(attachmentCount, attachments.data()));

            CHECK_GL(glGenFramebuffers(1, &fbo));
            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, fbo));

            counter = 0;
            auto initTex = [width, height, &counter](GLuint &tex)
            {
                CHECK_GL(glGenTextures(1, &tex));
                CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, tex));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
                CHECK_GL(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr));
                CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + counter, GL_TEXTURE_RECTANGLE, tex, 0));
                ++counter;
            };

            initTex(colorTex);
            initTex(positionTex);
            initTex(normalTex);
            initTex(texcoordTex);
            initTex(tangentTex);

            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, GL_NONE));

            CHECK_GL(glDrawBuffers(attachmentCount, attachments.data()));

            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE));

            isInit = true;
        }

        void release()
        {
            if (!isInit)
            {
                return;
            }

            CHECK_GL(glDeleteFramebuffers(1, &msFbo));
            CHECK_GL(glDeleteFramebuffers(1, &fbo));
            CHECK_GL(glDeleteRenderbuffers(1, &depthRbo));
            CHECK_GL(glDeleteRenderbuffers(1, &colorRbo));
            CHECK_GL(glDeleteTextures(1, &colorTex));
            CHECK_GL(glDeleteRenderbuffers(1, &texcoordRbo));
            CHECK_GL(glDeleteTextures(1, &texcoordTex));

            isInit = false;
        }

        void beforeDraw()
        {
            CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, msFbo));
        }

        void afterDraw(GLsizei width, GLsizei height)
        {
            for (unsigned int i = 0; i < attachmentCount; ++i)
            {
                CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, msFbo));
                CHECK_GL(glReadBuffer(GL_COLOR_ATTACHMENT0 + i));
                CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo));
                CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0 + i));
                CHECK_GL(glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
            }
        }

    }; // struct MRT

}; // namespace zenvis

#endif // __MRT__H__
