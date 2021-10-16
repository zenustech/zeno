
/** 
* @file notepad.cpp Demo 
*
* @brief Creating A Simple Notepad In Nana 0.8 (updated to 1.3 Alpha)
* 
* Let's start a tutorial to create a simple notepad, the simple notepad is a text editor 
* that allows the user to select and browse existing text files.
* This example also shows how you can use place, menubar, menu, textbox, msgbox, and filebox and their various options.
*
* This example requires Nana 1.3 Alpha for C++11 and a C++11 compiler.
* 
* Get Started
* 
* First of all, the whole program code is ready, and then we will go through each function.
*/


#include <nana/gui.hpp>
#include <nana/gui/widgets/menubar.hpp>
#include <nana/gui/widgets/textbox.hpp>
#include <nana/gui/place.hpp>
#include <nana/gui/msgbox.hpp>
#include <nana/gui/filebox.hpp>
#include <thread>
#include <iostream>


using namespace nana;

class notepad_form     : public form
{
    place   place_  {*this};
    menubar menubar_{*this};
    textbox textbox_{*this};

public:
    notepad_form()
    {
        caption("Simple Notepad - Nana C++ Library");
        textbox_.borderless(true);
        API::effects_edge_nimbus(textbox_, effects::edge_nimbus::none);
        textbox_.enable_dropfiles(true);
        textbox_.events().mouse_dropfiles([this](const arg_dropfiles& arg)
        {
            if (arg.files.size() && _m_ask_save())
                textbox_.load(arg.files.front());
        });

        _m_make_menus();

        place_.div("vert<menubar weight=28><textbox>");
        place_["menubar"] << menubar_;
        place_["textbox"] << textbox_;
        place_.collocate();

        events().unload([this](const arg_unload& arg){
            if (!_m_ask_save())
                arg.cancel = true;
        });
    }

    textbox& get_tb(){return textbox_;}
private:
    std::filesystem::path _m_pick_file(bool is_open) const
    {
        filebox fbox(*this, is_open);
        fbox.add_filter("Text", "*.txt");
        fbox.add_filter("All Files", "*.*");

        auto files = fbox.show();
	    return (files.empty() ? std::filesystem::path{} : files.front());
    }

    bool _m_ask_save()
    {
        if (textbox_.edited())
        {
            auto fs = textbox_.filename();
            msgbox box(*this, "Simple Notepad", msgbox::button_t::yes_no_cancel);
            box << "Do you want to save these changes?";

            switch (box.show())
            {
            case msgbox::pick_yes:
                if (fs.empty())
                {
                    fs = _m_pick_file(false);
                    if (fs.empty())
                        break;
                    if (fs.extension().string() != ".txt")
                        fs = fs.extension().string() + ".txt";
                }
                textbox_.store(fs);
                break;
            case msgbox::pick_cancel:
                return false;
            default:
                break;
            }
        }
        return true;
    }

    void _m_make_menus()
    {
        menubar_.push_back("&FILE");
        menubar_.at(0).append("New", [this](menu::item_proxy& ip)
        {
            if(_m_ask_save())
                textbox_.reset();
        });
        menubar_.at(0).append("Open", [this](menu::item_proxy& ip)
        {
            if (_m_ask_save())
            {
                auto fs = _m_pick_file(true);
                if (!fs.empty())
                    textbox_.load(fs);
            }
        });
        menubar_.at(0).append("Save", [this](menu::item_proxy&)
        {
            auto fs = textbox_.filename();
            if (fs.empty())
            {
                fs = _m_pick_file(false);
                if (fs.empty())
                    return;
            }
            textbox_.store(fs);
        });

        menubar_.push_back("F&ORMAT");
        menubar_.at(1).append("Line Wrap", [this](menu::item_proxy& ip)
        {
            textbox_.line_wrapped(ip.checked());
        });
        menubar_.at(1).check_style(0, menu::checks::highlight);
    }

};
void Wait(unsigned wait=0)
{
    if (wait)
        std::this_thread::sleep_for(std::chrono::seconds{ wait } );
}

int main()
{
    notepad_form npform;
    npform.show();
    exec(

#ifdef NANA_AUTOMATIC_GUI_TESTING
		1,1, [&npform]()
    {
        /*
        arg_keyboard k;
        k.shift=k.ctrl=false;
        k.evt_code=event_code::key_char;
        k.window_handle = npform.get_tb().handle();
        for (char c : nana::to_nstring( "Testing our notepad"))
        {
            k.key=c;
            std::cout<<c;
            npform.get_tb().events().key_char.emit(k); Wait(1);
        }
        */
    }
#endif

	);
}


