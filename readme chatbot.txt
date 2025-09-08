when download the python remember to tick the path Add Python 3.10 to PATH
just using the default download no need to customize 
after that downloaded the python 
open window type cmd
first check the version by using this command (py -3.10 --version)
make sure the version is Python 3.10.11
now need to check where is ur file download at (cd C:\Users\USER\Downloads\a1)
after that use this command to install (py -3.10 -m pip install numpy torch nltk)
one finished use this command (py -3.10 train.py)
u will saw a command is Training complete.
now u can put this command and get start to chat (py -3.10 app.py)
wheater (pip install requests pillow torch nltk)
https://www.weatherapi.com/my/
acc: weing924@gmail.com
pass: 123456aBc.



Step 1: Install Python 3.10
can just copy the link and paste it at website download Python 3.10.11 from the official website:
https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe
During installation:
Make sure to check the box "Add Python 3.10 to PATH"
Proceed with default installation settings

Step 2: Open Command Prompt
Press Win + R, type cmd, and press Enter
Verify that Python 3.10 is installed using the following command:
py -3.10 --version
Expected output:
Python 3.10.11

Step 3: Navigate to Project Folder
Ensure your project files are located in the following directory:
C:\Users\USER\Downloads\a1
Navigate to the directory:
cd C:\Users\USER\Downloads\a1

Step 4: Install Required Python Packages
Use the following command to install the required libraries:
py -3.10 -m pip install numpy torch nltk

Step 5: Train the Chatbot Model
Once the dependencies are installed, run the training script:
py -3.10 train.py
After successful training, you should see the message:
Training complete.
Model saved to data.pth

Step 6: Launch the Chatbot GUI
To start the chatbot interface, run:
py -3.10 app.py
A GUI window will appear where you can begin chatting with the bot.
