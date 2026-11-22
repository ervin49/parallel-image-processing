#include "ImageDisplayer.h"

void ImageDisplayer::uploadTexture(const cv::Mat& image, GLuint& texture)
{
    const cv::Mat source = image.isContinuous() ? image : image.clone();

    if (texture != 0)
        glDeleteTextures(1, &texture);
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    if (source.channels() == 1)
    {
        constexpr GLint swizzle[4] = {GL_RED, GL_RED, GL_RED, GL_ONE};
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA, swizzle);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, source.cols, source.rows, 0, GL_RED, GL_UNSIGNED_BYTE, source.data);
    }
    else if (source.channels() == 4)
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, source.cols, source.rows, 0, GL_RGBA, GL_UNSIGNED_BYTE, source.data);
    }
    else
    {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, source.cols, source.rows, 0, GL_RGB, GL_UNSIGNED_BYTE, source.data);
    }
}
