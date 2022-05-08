// MRT = Multiple Render targets
#ifndef __MRT_H__
#define __MRT_H__

#include "stdafx.hpp"
#include <iostream>

namespace zenvis
{
    struct MRT
    {
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

        bool is_use{false};
        bool is_init{false};

        static constexpr unsigned int attachment_count{6};
        GLuint fbo{GL_NONE};
        GLuint ms_fbo{GL_NONE};
        GLuint depth_rbo{GL_NONE};
        GLuint rbos[attachment_count];
        GLuint texs[attachment_count];

        /*
        fColor
        mrt_attr_pos
        mrt_attr_clr
        mrt_attr_nrm
        mrt_attr_uv
        mrt_attr_tang
        */

        void init(GLsizei samples, GLsizei width, GLsizei height)
        {
            if (is_init)
            {
                return;
            }

            CHECK_GL(glGenFramebuffers(1, &ms_fbo));
            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, ms_fbo));

            CHECK_GL(glGenRenderbuffers(1, &depth_rbo));
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, depth_rbo));
            CHECK_GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_DEPTH_COMPONENT32F, width, height));
            CHECK_GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rbo));

            CHECK_GL(glGenRenderbuffers(attachment_count, rbos));
            for (unsigned int i = 0; i < attachment_count; ++i)
            {
                CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, rbos[i]));
                CHECK_GL(glRenderbufferStorageMultisample(GL_RENDERBUFFER, samples, GL_RGBA32F, width, height));
                CHECK_GL(glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_RENDERBUFFER, rbos[i]));
            }
            CHECK_GL(glBindRenderbuffer(GL_RENDERBUFFER, GL_NONE));

            GLenum attachments[attachment_count];
            for (unsigned int i = 0; i < attachment_count; ++i)
            {
                attachments[i] = GL_COLOR_ATTACHMENT0 + i;
            }
            CHECK_GL(glDrawBuffers(attachment_count, attachments));

            CHECK_GL(glGenFramebuffers(1, &fbo));
            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, fbo));

            CHECK_GL(glGenTextures(attachment_count, texs));
            for (unsigned int i = 0; i < attachment_count; ++i)
            {
                CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, texs[i]));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MIN_FILTER, GL_NEAREST));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_MAG_FILTER, GL_NEAREST));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE));
                CHECK_GL(glTexParameteri(GL_TEXTURE_RECTANGLE, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE));
                CHECK_GL(glTexImage2D(GL_TEXTURE_RECTANGLE, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, nullptr));
                CHECK_GL(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0 + i, GL_TEXTURE_RECTANGLE, texs[i], 0));
            }
            CHECK_GL(glBindTexture(GL_TEXTURE_RECTANGLE, GL_NONE));

            CHECK_GL(glDrawBuffers(attachment_count, attachments));

            CHECK_GL(glBindFramebuffer(GL_FRAMEBUFFER, GL_NONE));

            is_init = true;
        }

        void release()
        {
            if (!is_init)
            {
                return;
            }

            CHECK_GL(glDeleteFramebuffers(1, &fbo));
            CHECK_GL(glDeleteFramebuffers(1, &ms_fbo));
            CHECK_GL(glDeleteRenderbuffers(1, &depth_rbo));
            CHECK_GL(glDeleteRenderbuffers(attachment_count, rbos));
            CHECK_GL(glDeleteTextures(attachment_count, texs));

            is_init = false;
        }

        void before_draw()
        {
            CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, ms_fbo));
        }

        void after_draw(GLsizei width, GLsizei height)
        {
            for (unsigned int i = 0; i < attachment_count; ++i)
            {
                CHECK_GL(glBindFramebuffer(GL_READ_FRAMEBUFFER, ms_fbo));
                CHECK_GL(glReadBuffer(GL_COLOR_ATTACHMENT0 + i));
                CHECK_GL(glBindFramebuffer(GL_DRAW_FRAMEBUFFER, fbo));
                CHECK_GL(glDrawBuffer(GL_COLOR_ATTACHMENT0 + i));
                CHECK_GL(glBlitFramebuffer(0, 0, width, height, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_NEAREST));
            }
        }

    }; // struct MRT

}; // namespace zenvis

#endif // __MRT__H__
