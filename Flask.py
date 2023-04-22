from flask import Flask, redirect

app = Flask(__name__)

redirect_link = 'https://2ae7c3224e0759a5c3.gradio.live/'

@app.route('/')
def index():
    return redirect(redirect_link)

if __name__ == '__main__':
    app.run(host='141.223.167.14', port=7172, debug=True)