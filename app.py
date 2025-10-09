import os
import io
import base64
import numpy as np
import json 
from flask import Flask, render_template, request, flash, redirect, url_for, send_file
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.core.credentials import AzureKeyCredential
import click

# --- Basic App Setup ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'a-super-secret-key-change-this' 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

# --- Database Configuration ---
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(basedir, 'assets.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Login Manager Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- Azure AI Configuration ---
try:
    AZURE_AI_ENDPOINT = os.environ['AZURE_AI_ENDPOINT']
    AZURE_AI_KEY = os.environ['AZURE_AI_KEY']
except KeyError:
    # If the keys aren't set, the app will fail fast, which is good for debugging.
    # You can add a more user-friendly error message here if you like.
    print("FATAL ERROR: Azure AI environment variables not set.")
    print("Please set AZURE_AI_ENDPOINT and AZURE_AI_KEY.")
    exit(1) # Exit the application if keys are missing

# --- Database Model Definitions ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False) # Increased length for stronger hashes
    role = db.Column(db.String(80), nullable=False, default='user')
    complaints = db.relationship('Complaint', backref='user', lazy=True, cascade="all, delete-orphan")

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class LostItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    location = db.Column(db.String(200), nullable=False)
    contact = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(50), nullable=False, default='Found')
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    images = db.relationship('ItemImage', backref='item', lazy=True, cascade="all, delete-orphan")

class ItemImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_data = db.Column(db.LargeBinary, nullable=False)
    item_id = db.Column(db.Integer, db.ForeignKey('lost_item.id'), nullable=False)
    vector = db.Column(db.Text) 

class Complaint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=False)
    location = db.Column(db.String(200), nullable=False)
    contact = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime(timezone=True), server_default=func.now())
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    images = db.relationship('ComplaintImage', backref='complaint', lazy=True, cascade="all, delete-orphan")

class ComplaintImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_data = db.Column(db.LargeBinary, nullable=False)
    complaint_id = db.Column(db.Integer, db.ForeignKey('complaint.id'), nullable=False)
    vector = db.Column(db.Text) 

# --- Login Manager User Loader ---
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- Helper Functions (Tokenizing, AI Analysis) ---
def tokenize_description(text):
    if not text: return set()
    clean_text = text.lower().replace('.', '').replace(',', '').replace("features:", "").replace("visible text:", "").replace("'", "")
    stop_words = {'a', 'an', 'the', 'is', 'it', 'and', 'or', 'such', 'as', 'of', 'in', 'that', 'with'}
    return set(word for word in clean_text.split() if word and word not in stop_words)

def find_similar_items_by_text(complaint_description):
    all_lost_items = LostItem.query.all()
    complaint_tokens = tokenize_description(complaint_description)
    matches = {}
    if not complaint_tokens: return []
    for item in all_lost_items:
        item_tokens = tokenize_description(item.description)
        intersection = len(complaint_tokens.intersection(item_tokens))
        union = len(complaint_tokens.union(item_tokens))
        similarity = intersection / union if union > 0 else 0
        if similarity > 0.2: 
            matches[item.id] = {'item': item, 'similarity': similarity}
    sorted_matches = sorted(matches.values(), key=lambda x: x['similarity'], reverse=True)
    return [match['item'] for match in sorted_matches]

def analyze_image_data(image_bytes):
    try:
        client = ImageAnalysisClient(endpoint=AZURE_AI_ENDPOINT, credential=AzureKeyCredential(AZURE_AI_KEY))
        result = client.analyze(image_data=image_bytes, visual_features=['Tags', 'Read'])
        description_parts = []
        if result.tags:
            top_tags = [tag.name for tag in result.tags.list if tag.confidence > 0.7]
            if top_tags: description_parts.append("Features: " + ", ".join(top_tags) + ".")
        if result.read and result.read.blocks:
            found_text = " ".join([line.text for block in result.read.blocks for line in block.lines])
            if found_text: description_parts.append("Visible text: '" + found_text + "'.")
        ai_description = " ".join(description_parts) if description_parts else "Could not generate a detailed description."
    except Exception as e:
        ai_description = f"AI Description Failed: ({e}). Using local fallback text."
        print(ai_description)
    vector_data = np.random.uniform(low=-1, high=1, size=(512,)).tolist() 
    return {'description': ai_description, 'vector': vector_data}

# --- Image Serving Routes ---
@app.route('/image/<int:image_id>')
def get_image(image_id):
    image = ItemImage.query.get_or_404(image_id)
    return send_file(io.BytesIO(image.image_data), mimetype='image/jpeg')

@app.route('/complaint_image/<int:image_id>')
@login_required
def get_complaint_image(image_id):
    image = ComplaintImage.query.get_or_404(image_id)
    return send_file(io.BytesIO(image.image_data), mimetype='image/jpeg')

# --- Core App Routes ---

# NEW: Public marketplace view
@app.route('/')
def marketplace():
    items = LostItem.query.order_by(LostItem.created_at.desc()).all()
    user_complaints_with_match_ids = {} 
    if current_user.is_authenticated:
        user_complaints = Complaint.query.filter_by(user_id=current_user.id).all()
        for complaint in user_complaints:
            matches = find_similar_items_by_text(complaint.description)
            if matches:
                user_complaints_with_match_ids[complaint.title] = [match.id for match in matches]
    return render_template(
        'marketplace.html', 
        items=items, 
        user_complaints_with_match_ids=user_complaints_with_match_ids
    )

@app.route('/home')
@login_required
def home():
    """Redirects logged-in users to the correct dashboard."""
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    else:
        # Regular users can now use the main marketplace as their homepage
        return redirect(url_for('marketplace'))

# RENAMED for clarity
@app.route('/admin')
@login_required
def admin_dashboard():
    """Admin-only dashboard."""
    if current_user.role != 'admin':
        flash('Access denied: Admins only.', 'danger')
        return redirect(url_for('marketplace'))
    items = LostItem.query.order_by(LostItem.created_at.desc()).all()
    # This renders the admin-specific view, now named 'admin_dashboard.html'
    # We will rename the old index.html to admin_dashboard.html
    return render_template('admin_dashboard.html', items=items)


@app.route('/admin/complaints')
@login_required
def view_complaints(complaint_id=None):
    if current_user.role != 'admin': return redirect(url_for('marketplace'))
    complaints = Complaint.query.order_by(Complaint.created_at.desc()).all()
    return render_template('complaints.html', 
                           complaints=complaints)
@app.route('/admin/item/edit/<int:item_id>', methods=['POST'])
@login_required
def edit_item(item_id):
    """Allows an admin to edit the details of a specific item."""
    if current_user.role != 'admin':
        flash('You do not have permission to perform this action.', 'danger')
        return redirect(url_for('marketplace'))

    item = LostItem.query.get_or_404(item_id)
    try:
        # Update item fields from the form data
        item.title = request.form['title']
        item.location = request.form['location']
        item.contact = request.form['contact']
        item.status = request.form['status']
        
        db.session.commit()
        flash(f'Item "{item.title}" has been successfully updated.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while updating the item: {e}', 'danger')

    return redirect(url_for('admin_dashboard'))

# NEW: Route for an admin to delete an item
@app.route('/admin/item/delete/<int:item_id>', methods=['POST'])
@login_required
def delete_item(item_id):
    """Allows an admin to permanently delete an item."""
    if current_user.role != 'admin':
        flash('You do not have permission to perform this action.', 'danger')
        return redirect(url_for('marketplace'))

    item = LostItem.query.get_or_404(item_id)
    try:
        # Cascade delete should handle associated images
        db.session.delete(item)
        db.session.commit()
        flash(f'Item "{item.title}" has been successfully deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while deleting the item: {e}', 'danger')
        
    return redirect(url_for('admin_dashboard'))
# NEW: Route for an admin to delete a user complaint
@app.route('/admin/complaint/delete/<int:complaint_id>', methods=['POST'])
@login_required
def delete_complaint(complaint_id):
    """Allows an admin to delete a specific complaint."""
    # Step 1: Ensure the current user is an admin
    if current_user.role != 'admin':
        flash('You do not have permission to perform this action.', 'danger')
        return redirect(url_for('marketplace'))

    # Step 2: Find the complaint in the database
    complaint = Complaint.query.get_or_404(complaint_id)
    
    try:
        # Step 3: Delete the complaint and commit the change
        db.session.delete(complaint)
        db.session.commit()
        flash(f'Complaint "{complaint.title}" has been successfully deleted.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'An error occurred while deleting the complaint: {e}', 'danger')

    # Step 4: Redirect back to the main complaints view
    return redirect(url_for('view_complaints'))

# This route is now mainly for admins to post new items
@app.route('/admin/analyze', methods=['POST']) 
@login_required 
def analyze_image(): 
    if current_user.role != 'admin':
        flash('Only admins can log new items.', 'danger')
        return redirect(url_for('marketplace'))

    title, location, contact, status = request.form['title'], request.form['location'], request.form['contact'], request.form['status']
    files = request.files.getlist('file')
    if not files or files[0].filename == '': 
        flash('At least one image is required.', 'danger') 
        return redirect(url_for('admin_dashboard'))
    
    try: 
        all_descriptions, vector_from_first_image = [], None
        for file in files:
            image_bytes = file.read()
            if not image_bytes: continue
            analysis = analyze_image_data(image_bytes)
            all_descriptions.append(analysis['description'])
            if vector_from_first_image is None: vector_from_first_image = analysis['vector']
            file.seek(0)
        
        unique_descriptions = set(d for d in all_descriptions if "Failed" not in d and "Could not generate" not in d)
        net_description = " | ".join(unique_descriptions) if unique_descriptions else "No detailed description could be generated."
        vector_data_str = json.dumps(vector_from_first_image) if vector_from_first_image else '[]'
        
        new_item = LostItem(title=title, description=net_description, location=location, contact=contact, status=status) 
        db.session.add(new_item) 
        db.session.commit() 
        for file in files: 
            image_bytes = file.read() 
            if image_bytes: db.session.add(ItemImage(image_data=image_bytes, item_id=new_item.id, vector=vector_data_str))
        db.session.commit() 
        flash('New item added and analyzed successfully!', 'success') 
        return redirect(url_for('admin_dashboard'))
    except Exception as e: 
        db.session.rollback() 
        flash(f'An error occurred: {e}', 'danger') 
        return redirect(url_for('admin_dashboard'))

# --- User-Specific Routes ---
@app.route('/my-matches')
@login_required
def my_matches():
    user_complaints = Complaint.query.filter_by(user_id=current_user.id).order_by(Complaint.created_at.desc()).all()
    complaints_with_matches = []
    for complaint in user_complaints:
        matches = find_similar_items_by_text(complaint.description)
        complaints_with_matches.append({'complaint': complaint, 'matches': matches})
    return render_template('my_matches.html', 
                           complaints_with_matches=complaints_with_matches,
                           complaints=user_complaints)

@app.route('/new-complaint', methods=['GET', 'POST']) 
@login_required 
def new_complaint(): 
    if request.method == 'POST': 
        title, location, contact = request.form['title'], request.form['location'], request.form['contact']
        files = request.files.getlist('file')
        if not files or files[0].filename == '': 
            flash('At least one image is required.', 'danger') 
            return redirect(request.url) 
        try: 
            analysis = analyze_image_data(files[0].read())
            complaint = Complaint(title=title, description=analysis['description'], location=location, contact=contact, user_id=current_user.id)
            db.session.add(complaint) 
            db.session.commit() 
            for file in files: 
                file.seek(0) 
                db.session.add(ComplaintImage(image_data=file.read(), complaint_id=complaint.id, vector=json.dumps(analysis['vector'])))
            db.session.commit() 
            flash('Your complaint has been submitted!', 'success') 
            return redirect(url_for('my_matches')) 
        except Exception as e: 
            db.session.rollback() 
            flash(f'An error occurred: {e}', 'danger') 
            return redirect(request.url) 
    return render_template('new_complaint.html') 

# --- Authentication Routes ---
@app.route('/login', methods=['GET', 'POST']) 
def login(): 
    if current_user.is_authenticated: return redirect(url_for('home')) 
    if request.method == 'POST': 
        user = User.query.filter_by(username=request.form['username']).first() 
        if user is None or not user.check_password(request.form['password']): 
            flash('Invalid username or password.', 'danger') 
            return redirect(url_for('login')) 
        login_user(user)
        return redirect(url_for('home')) 
    return render_template('login.html') 

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Please choose a different one.', 'danger')
            return redirect(url_for('signup'))
        new_user = User(username=username)
        new_user.set_password(password)
        db.session.add(new_user)
        db.session.commit()
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('signup.html')


@app.route('/logout') 
@login_required
def logout(): 
    logout_user() 
    return redirect(url_for('marketplace')) 

# --- CLI Commands ---
@app.cli.command('init-db') 
def init_db_command(): 
    with app.app_context(): 
        db.drop_all()
        db.create_all() 
    print('Initialized and reset the database.') 

@app.cli.command('create-admin')
def create_admin_command():
    with app.app_context():
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(username='admin', role='admin')
            admin.set_password('admin')
            db.session.add(admin)
            db.session.commit()
            print('Admin user "admin" created (password: admin).')
        else:
            print('Admin user already exists.')

@app.cli.command('create-user')
@click.argument('username')
@click.argument('password')
def create_user_command(username, password):
    with app.app_context():
        user = User.query.filter_by(username=username).first()
        if not user:
            user = User(username=username, role='user')
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            print(f'User "{username}" created.')
        else:
            print(f'User "{username}" already exists.')


if __name__ == '__main__': 
    app.run(debug=True, host='0.0.0.0')