from flask import Flask, render_template, request, redirect

app = Flask(__name__)

@app.route('/contact', methods=['POST'])
def contact():
  name = request.form['name']
  email = request.form['email']
  message = request.form['message']

  # Send an email with the form data
  send_email(name, email, message)

  # Redirect to the home page
  return redirect('/')

if __name__ == '__main__':
  app.run(debug=True)