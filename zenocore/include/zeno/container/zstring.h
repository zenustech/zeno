#pragma once

#ifndef __ZSTRING_H__
#define __ZSTRING_H__

#include <string.h>
#include <cstdlib>

namespace zeno
{
    class String
    {
    public:
        String() noexcept : m_data(0), m_length(0) {}

        String(const char* str) noexcept 
            : m_data(0), m_length(0) {
            m_length = _strlen(str);
            m_data = new char[m_length + 1];
            _strcpy(m_data, str);
        }

        String(const String& rhs) noexcept 
            : m_data(0), m_length(0) {
            m_length = _strlen(rhs.m_data);
            if (m_length > 0) {
                m_data = new char[m_length + 1];
                _strcpy(m_data, rhs.m_data);
            }
        }

        String(String&& rhs) noexcept {
            m_data = rhs.m_data;
            m_length = rhs.m_length;
            rhs.m_data = nullptr;
            rhs.m_length = 0;
        }

        ~String() {
            free(m_data);
            m_length = 0;
        }

        size_t length() const noexcept {
            return m_length;
        }

        size_t size() const noexcept {
            return m_length;
        }

        bool empty() const noexcept {
            return m_length == 0;
        }

        char operator[](size_t idx) noexcept {
            return *(m_data + idx);
        }

        const char* data() const noexcept {
            return m_data;
        }

        const char* c_str() const noexcept {
            return m_data;
        }

        String& operator=(const char* str) noexcept {
            free(m_data);
            m_data = nullptr;
            m_length = _strlen(str);
            if (m_length > 0) {
                m_data = new char[m_length + 1];
                _strcpy(m_data, str);
            }
            return *this;
        }

        String& operator=(const String& other) {
            if (this != &other) {
                delete[] m_data;
                m_data = nullptr;
                m_length = _strlen(other.m_data);
                if (m_length > 0) {
                    m_data = new char[m_length + 1];
                    _strcpy(m_data, other.m_data);
                }
            }
            return *this;
        }

        bool operator==(const String& rhs) const noexcept {
            return strcmp(m_data, rhs.m_data) == 0;
        }

        bool operator<(const String& rhs) const {
            return strcmp(m_data, rhs.m_data) < 0;
        }

        String operator+(const char c) const noexcept {
            size_t newLen = m_length + 1;
            char* newData = new char[newLen + 1];
            _strcpy(newData, m_data);
            newData[m_length] = c;
            newData[newLen] = '\0';
            String result(newData);
            delete[] newData;
            return result;
        }

        String operator+(const char* str) const noexcept {
            if (!str) return *this;
            size_t newLen = m_length + _strlen(str);
            char* newData = new char[newLen + 1];
            _strcpy(newData, m_data);
            _strcat(newData, str);
            String result(newData);
            delete[] newData;
            return result;
        }

        String operator+(const String& rhs) const noexcept {
            if (rhs.m_length == 0)
                return *this;
            size_t newLen = _strlen(m_data) + _strlen(rhs.m_data);
            char* newData = new char[newLen + 1];
            _strcpy(newData, m_data);
            _strcat(newData, rhs.m_data);
            String result(newData);
            delete[] newData;
            return result;
        }

        // 友元函数重载：const char* + String
        friend String operator+(const char* lhs, const String& rhs) {
            size_t newLen = _strlen(lhs) + _strlen(rhs.m_data);
            char* newData = new char[newLen + 1];
            _strcpy(newData, lhs);
            _strcat(newData, rhs.m_data);
            String result(newData);
            delete[] newData;
            return result;
        }

        friend String operator+(const char c, const String& rhs) {
            size_t newLen = 1 + _strlen(rhs.m_data);
            char* newData = new char[newLen + 1];
            newData[0] = c;
            _strcpy(newData + 1, rhs.m_data);
            String result(newData);
            delete[] newData;
            return result;
        }

    private:
        static char* _strcpy(char* dest, const char* src) {
            char* original = dest;
            while ((*dest++ = *src++) != '\0');
            return original;
        }

        static char* _strcat(char* dest, const char* src) {
            char* original = dest;
            while (*dest) {
                dest++;
            }
            while ((*dest++ = *src++) != '\0');
            return original;
        }

        static size_t _strlen(const char* str) noexcept {
            if (!str) return 0;
            size_t len = 0;
            while (*str++ != '\0') len++;
            return len;
        }

        char* m_data;
        size_t m_length;
    };
}


#endif