// =============================================
// SECTION: Core Configuration & Initialization
// =============================================

/**
 * Application configuration constants
 * @type {Object}
 */
const CONFIG = {
  // Animation and UI settings
  animationDuration: 800,
  scrollThreshold: 0.2,
  preloaderDuration: 3000,
  maxPreloaderTimeout: 10000,
  
  // XP and progression system
  xpValues: {
    correctAnswer: 100,
    incorrectAnswer: -20,
    streakBonus: 50,
    levelMultiplier: 1000,
    quizCompletion: 200,
    algorithmView: 20,
    projectCompletion: 500
  },
  
  // Local storage keys
  storageKeys: {
    theme: 'datakarya_theme',
    progress: 'datakarya_progress',
    achievements: 'datakarya_achievements',
    userSettings: 'datakarya_settings',
    userProjects: 'datakarya_projects'
  },
  
  // Color themes
  colorThemes: {
    default: {
      primary: '#6e48aa',
      secondary: '#9d50bb',
      accent: '#4776E6',
      text: '#333333',
      background: '#ffffff'
    },
    terminal: {
      primary: '#0a0e12',
      secondary: '#1f2937',
      accent: '#2ec866',
      text: '#e0e0e0',
      background: '#121212'
    },
    ocean: {
      primary: '#1a2980',
      secondary: '#26d0ce',
      accent: '#00c9ff',
      text: '#f5f5f5',
      background: '#051a38'
    },
    forest: {
      primary: '#1e3f20',
      secondary: '#4a7856',
      accent: '#94c356',
      text: '#e8f5e9',
      background: '#0d1f12'
    }
  },
  
  // Algorithm categories
  categories: {
    supervised: 'Supervised Learning',
    unsupervised: 'Unsupervised Learning',
    reinforcement: 'Reinforcement Learning',
    deep: 'Deep Learning',
    ensemble: 'Ensemble Methods',
    dimensionality: 'Dimensionality Reduction'
  },
  
  // Difficulty levels
  difficulties: {
    beginner: { name: 'Beginner', color: '#4CAF50' },
    intermediate: { name: 'Intermediate', color: '#2196F3' },
    advanced: { name: 'Advanced', color: '#9C27B0' },
    expert: { name: 'Expert', color: '#FF5722' }
  },
  
  // Playground datasets
  playgroundDatasets: {
    classification: ['Iris', 'MNIST', 'CIFAR-10', 'Breast Cancer', 'Wine Quality'],
    regression: ['Boston Housing', 'Diabetes', 'California Housing', 'Energy Efficiency'],
    clustering: ['Mall Customers', 'Wholesale Customers', 'Old Faithful', 'Iris (Unsupervised)']
  },
  
  // Default hyperparameters for algorithms
  defaultHyperparams: {
    'linear-regression': {
      fit_intercept: true,
      normalize: false
    },
    'logistic-regression': {
      penalty: 'l2',
      C: 1.0,
      solver: 'lbfgs'
    },
    'decision-tree': {
      max_depth: 3,
      min_samples_split: 2
    },
    'random-forest': {
      n_estimators: 100,
      max_depth: 3
    },
    'svm': {
      C: 1.0,
      kernel: 'rbf'
    },
    'kmeans': {
      n_clusters: 3,
      init: 'k-means++'
    },
    'neural-network': {
      hidden_layer_sizes: [100],
      activation: 'relu',
      learning_rate: 'constant'
    }
  }
};

/**
 * DOM element references
 * @type {Object}
 */
const DOM = {
  // Preloader elements
  preloader: null,
  progressBar: null,
  
  // Navigation elements
  navbar: null,
  menuToggle: null,
  navLinks: null,
  navLinksList: null,
  
  // Search elements
  searchInput: null,
  searchBtn: null,
  
  // Algorithm elements
  algorithmGrid: null,
  algorithmCards: null,
  algorithmDetailsContainer: null,
  
  // Filter elements
  filterBtns: null,
  difficultyBtns: null,
  
  // Section elements
  heroSection: null,
  dashboardSection: null,
  algorithmsSection: null,
  roadmapSection: null,
  resourcesSection: null,
  contactSection: null,
  playgroundSection: null,
  projectsSection: null,
  
  // Footer elements
  footer: null,
  
  // Other interactive elements
  contactForm: null,
  timelineItems: null,
  playgroundForm: null,
  projectForm: null
};

/**
 * Application state management
 * @type {Object}
 */
const STATE = {
  // UI state
  isMenuOpen: false,
  isDarkMode: false,
  currentTheme: 'default',
  
  // Algorithm state
  currentAlgorithm: null,
  algorithmFilter: 'all',
  difficultyFilter: 'all',
  currentTab: 'visualization',
  currentCodeLang: 'python',
  
  // Playground state
  playground: {
    activeAlgorithm: null,
    activeDataset: null,
    hyperparameters: {},
    visualizationType: '2d',
    isTraining: false,
    trainingProgress: 0,
    results: null
  },
  
  // Projects state
  userProjects: [],
  currentProject: null,
  
  // User progress
  quizProgress: {},
  userAchievements: [],
  completedAlgorithms: [],
  completedProjects: [],
  xp: 0,
  level: 1,
  streak: 0,
  streakDate: null,
  
  // Navigation state
  scrollPosition: 0,
  lastScrollPosition: 0,
  scrollDirection: 'down',
  activeSection: null,
  
  // Easter eggs
  konamiCode: [],
  konamiSequence: [
    'ArrowUp', 'ArrowUp', 'ArrowDown', 'ArrowDown', 
    'ArrowLeft', 'ArrowRight', 'ArrowLeft', 'ArrowRight', 
    'b', 'a'
  ],
  
  // Initialization flags
  isInitialized: false,
  resourcesLoaded: false
};

// =============================================
// SECTION: Initialization Functions
// =============================================

/**
 * Main initialization function
 */
function initializeApp() {
  // Initialize DOM references
  initializeDOMReferences();
  
  // Only proceed if essential elements exist
  if (!DOM.preloader || !DOM.progressBar) {
    console.error('Critical elements missing - falling back to basic initialization');
    DOM.preloader.style.display = 'none';
    completeInitialization();
    return;
  }
  
  // Initialize preloader
  initPreloader();
  
  // Set up mutation observer for dynamic content
  setupMutationObserver();
  
  // Set initialization flag
  STATE.isInitialized = true;

  // In main.js, where you initialize the app
  // First check if ALGORITHMS is loaded
  if (!ALGORITHMS || ALGORITHMS.length === 0) {
    console.error('ALGORITHMS data not loaded!');
    return;
  }
  generateAlgorithmCards(); // This should now work
}

/**
 * Initialize all DOM references
 */
function initializeDOMReferences() {
  // Preloader elements
  DOM.preloader = document.getElementById('preloader');
  DOM.progressBar = document.querySelector('.preloader .progress');
  
  // Navigation elements
  DOM.navbar = document.querySelector('.cinematic-navbar');
  DOM.menuToggle = document.querySelector('.menu-toggle');
  DOM.navLinks = document.querySelector('.nav-links');
  DOM.navLinksList = document.querySelector('.nav-links ul');
  
  // Search elements
  DOM.searchInput = document.querySelector('.search-input');
  DOM.searchBtn = document.querySelector('.search-btn');
  
  // Algorithm elements
  DOM.algorithmGrid = document.getElementById('algorithmGrid');
  DOM.algorithmCards = document.querySelectorAll('.algorithm-card');
  DOM.algorithmDetailsContainer = document.getElementById('algorithmDetailsContainer');
  
  // Filter elements
  DOM.filterBtns = document.querySelectorAll('.filter-btn');
  DOM.difficultyBtns = document.querySelectorAll('.difficulty-btn');
  
  // Section elements
  DOM.heroSection = document.querySelector('.hero-section');
  DOM.dashboardSection = document.getElementById('dashboard');
  DOM.algorithmsSection = document.getElementById('algorithms');
  DOM.roadmapSection = document.getElementById('roadmap');
  DOM.resourcesSection = document.getElementById('resources');
  DOM.contactSection = document.getElementById('contact');
  DOM.playgroundSection = document.getElementById('playground');
  DOM.projectsSection = document.getElementById('projects');
  
  // Footer elements
  DOM.footer = document.querySelector('.main-footer');
  
  // Other interactive elements
  DOM.contactForm = document.getElementById('contactForm');
  DOM.playgroundForm = document.getElementById('playgroundForm');
  DOM.projectForm = document.getElementById('projectForm');
  DOM.timelineItems = document.querySelectorAll('.timeline-item');
}

/**
 * Initialize preloader animation
 */
function initPreloader() {
  let progress = 0;
  const increment = 1;
  const interval = CONFIG.preloaderDuration / 100;
  
  const animateProgress = () => {
    progress += increment;
    DOM.progressBar.style.width = `${progress}%`;
    
    if (progress < 100) {
      setTimeout(animateProgress, interval);
    } else {
      transitionToContent();
    }
  };
  
  // Start animation
  setTimeout(animateProgress, interval);
}

/**
 * Handle transition from preloader to main content
 */
function transitionToContent() {
  DOM.preloader.style.opacity = '0';
  
  // Use both transitionend and fallback timeout
  let transitionEnded = false;
  
  const onTransitionEnd = () => {
    if (transitionEnded) return;
    transitionEnded = true;
    DOM.preloader.style.display = 'none';
    completeInitialization();
  };
  
  DOM.preloader.addEventListener('transitionend', onTransitionEnd);
  
  // Fallback in case transitionend doesn't fire
  setTimeout(onTransitionEnd, 1000);
}

/**
 * Complete application initialization
 */
function completeInitialization() {
  // Load user data and preferences
  loadUserData();
  
  // Generate dynamic content
  generateAlgorithmCards();
  generateAlgorithmDetails();
  initPlayground();
  initProjects();
  
  // Initialize event listeners
  initEventListeners();
  
  // Initialize animations and effects
  initAnimations();
  
  // Check for first-time user
  checkFirstTimeUser();
  
  // Set up service worker if available
  setupServiceWorker();
}

/**
 * Set up MutationObserver for dynamic content
 */
function setupMutationObserver() {
  const observer = new MutationObserver((mutations) => {
    mutations.forEach((mutation) => {
      if (mutation.type === 'childList') {
        // Reinitialize relevant components when DOM changes
        initializeDOMReferences();
        initEventListeners();
      }
    });
  });
  
  observer.observe(document.body, {
    childList: true,
    subtree: true
  });
}

// =============================================
// SECTION: Event Handling
// =============================================

/**
 * Initialize all event listeners
 */
function initEventListeners() {
  // Navigation events
  initNavigationEvents();
  
  // Search events
  initSearchEvents();
  
  // Algorithm events
  initAlgorithmEvents();
  
  // Filter events
  initFilterEvents();
  
  // Quiz events
  initQuizEvents();
  
  // Contact form events
  initContactFormEvents();
  
  // Playground events
  initPlaygroundEvents();
  
  // Project events
  initProjectEvents();
  
  // Scroll and resize events
  initScrollEvents();
  
  // Keyboard shortcuts
  initKeyboardShortcuts();
  
  // Theme and UI events
  initUIEvents();
}

/**
 * Initialize navigation event listeners
 */
function initNavigationEvents() {
  // Menu toggle
  if (DOM.menuToggle) {
    DOM.menuToggle.addEventListener('click', () => toggleMenu());
  }
  
  // Nav link clicks
  if (DOM.navLinksList) {
    DOM.navLinksList.addEventListener('click', (e) => {
      if (e.target.classList.contains('nav-link')) {
        toggleMenu(false);
        smoothScrollToSection(e.target.getAttribute('href'));
      }
    });
  }
  
  // Document click handler for closing menu
  document.addEventListener('click', (e) => {
    if (STATE.isMenuOpen && !e.target.closest('.navbar-container')) {
      toggleMenu(false);
    }
  });
}

/**
 * Initialize search event listeners
 */
function initSearchEvents() {
  if (DOM.searchInput && DOM.searchBtn) {
    // Debounced search input
    DOM.searchInput.addEventListener('input', debounce(handleSearch, 300));
    
    // Search button click
    DOM.searchBtn.addEventListener('click', handleSearchClick);
    
    // Search keyboard shortcuts
    DOM.searchInput.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        DOM.searchInput.value = '';
        handleSearch();
      }
    });
  }
}

/**
 * Initialize algorithm event listeners
 */
function initAlgorithmEvents() {
  // Algorithm card interactions
  document.querySelectorAll('.algorithm-card').forEach((card) => {
    card.addEventListener('mouseenter', () => card.classList.add('hovered'));
    card.addEventListener('mouseleave', () => card.classList.remove('hovered'));
    
    // Explore button
    const exploreBtn = card.querySelector('.explore-btn');
    if (exploreBtn) {
      exploreBtn.addEventListener('click', (e) => {
        const algorithm = e.currentTarget.dataset.algorithm;
        showAlgorithmDetail(algorithm);
      });
    }
    
    // Code button
    const codeBtn = card.querySelector('.code-btn');
    if (codeBtn) {
      codeBtn.addEventListener('click', (e) => {
        const algorithm = e.currentTarget.dataset.algorithm;
        showCodePreview(algorithm, e.currentTarget.closest('.algorithm-card'));
      });
    }
  });
  
  // Algorithm detail interactions
  document.querySelectorAll('.close-detail').forEach((btn) => {
    btn.addEventListener('click', closeAlgorithmDetail);
  });
  
  // Tab switching in algorithm details
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', handleTabSwitch);
  });
  
  // Code language switching
  document.querySelectorAll('.code-tab-btn').forEach((btn) => {
    btn.addEventListener('click', handleCodeTabSwitch);
  });
  
  // Algorithm navigation (next/prev)
  document.querySelectorAll('.prev-algorithm, .next-algorithm').forEach((btn) => {
    btn.addEventListener('click', navigateAlgorithm);
  });
}

/**
 * Initialize filter event listeners
 */
function initFilterEvents() {
  // Category filter buttons
  DOM.filterBtns.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const filterValue = e.currentTarget.dataset.filter;
      handleFilterChange(filterValue, STATE.difficultyFilter);
    });
  });
  
  // Difficulty filter buttons
  DOM.difficultyBtns.forEach((btn) => {
    btn.addEventListener('click', (e) => {
      const difficultyValue = e.currentTarget.dataset.difficulty;
      handleFilterChange(STATE.algorithmFilter, difficultyValue);
    });
  });
}

// =============================================
// SECTION: Filtering & Sorting
// =============================================

/**
 * Handle filter changes
 * @param {string} categoryFilter - Selected category filter
 * @param {string} difficultyFilter - Selected difficulty filter
 */
function handleFilterChange(categoryFilter, difficultyFilter) {
  // Update state
  STATE.algorithmFilter = categoryFilter;
  STATE.difficultyFilter = difficultyFilter;
  
  // Update active button states
  updateFilterButtonStates();
  
  // Apply filters
  applyAlgorithmFilters();
}

/**
 * Update active filter button states
 */
function updateFilterButtonStates() {
  // Update category filter buttons
  DOM.filterBtns.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.filter === STATE.algorithmFilter);
  });
  
  // Update difficulty filter buttons
  DOM.difficultyBtns.forEach((btn) => {
    btn.classList.toggle('active', btn.dataset.difficulty === STATE.difficultyFilter);
  });
}

/**
 * Apply current filters to algorithm cards
 */
function applyAlgorithmFilters() {
  const algorithmCards = document.querySelectorAll('.algorithm-card');
  
  algorithmCards.forEach((card) => {
    const cardCategory = card.dataset.category;
    const cardDifficulty = card.dataset.difficulty;
    
    // Check if card matches both filters
    const categoryMatch = STATE.algorithmFilter === 'all' || 
                         cardCategory === STATE.algorithmFilter ||
                         (STATE.algorithmFilter === 'deep-learning' && cardCategory === 'deep');
    
    const difficultyMatch = STATE.difficultyFilter === 'all' || 
                           cardDifficulty === STATE.difficultyFilter;
    
    if (categoryMatch && difficultyMatch) {
      card.style.display = 'block';
      card.classList.add('filter-match');
      card.classList.remove('filter-mismatch');
    } else {
      card.style.display = 'none';
      card.classList.add('filter-mismatch');
      card.classList.remove('filter-match');
    }
  });
  
  // Add animation for filtering
  animateFilterTransition();
}

/**
 * Animate filter transition
 */
function animateFilterTransition() {
  const grid = document.querySelector('.algorithm-grid');
  grid.style.opacity = '0.5';
  grid.style.transition = 'opacity 0.2s ease';
  
  setTimeout(() => {
    grid.style.opacity = '1';
  }, 200);
}

/**
 * Initialize quiz event listeners
 */
function initQuizEvents() {
  // Quiz submission
  document.querySelectorAll('.submit-quiz').forEach((btn) => {
    btn.addEventListener('click', handleQuizSubmit);
  });
  
  // Quiz reset
  document.querySelectorAll('.reset-quiz').forEach((btn) => {
    btn.addEventListener('click', resetQuiz);
  });
}

/**
 * Initialize contact form event listeners
 */
// Initialize form event listeners
function initContactFormEvents() {
  const contactForm = document.getElementById('contactForm');
  
  if (contactForm) {
    // Handle form submission
    contactForm.addEventListener('submit', handleContactSubmit);

    // Add validation to form fields on blur event
    contactForm.querySelectorAll('input, textarea, select').forEach((field) => {
      field.addEventListener('blur', validateFormField);
    });
  }
}

// Handle form submission
function handleContactSubmit(event) {
  event.preventDefault(); // Prevent the default form submission behavior
  
  const contactForm = event.target;
  
  // Validate the form before proceeding
  if (!validateForm(contactForm)) {
    return; // If validation fails, stop form submission
  }
  
  // Collect form data
  const formData = new FormData(contactForm);

  // Send formData to Formspree (or other backend)
  fetch(contactForm.action, {
    method: contactForm.method,
    body: formData,
  })
    .then((response) => {
      if (response.ok) {
        alert('Thank you! Your message has been submitted.');
        contactForm.reset(); // Reset the form on successful submission
      } else {
        alert('Oops! Something went wrong, please try again later.');
      }
    })
    .catch(() => {
      alert('There was an error while submitting the form. Please try again.');
    });
}

// Validate form fields (before submitting)
function validateForm(form) {
  let isValid = true;

  // Loop through each field to check for validity
  form.querySelectorAll('input, textarea, select').forEach((field) => {
    if (!validateFormField({ target: field })) {
      isValid = false;
    }
  });

  return isValid;
}

// Validate a single form field
function validateFormField(event) {
  const field = event.target;
  const value = field.value.trim();
  const errorMessageElement = field.nextElementSibling;

  // Remove any existing error messages
  if (errorMessageElement && errorMessageElement.classList.contains('error-message')) {
    errorMessageElement.remove();
  }

  // Check if the field is empty or invalid
  if (!value && field.hasAttribute('required')) {
    field.classList.add('error');
    
    const errorMsg = document.createElement('span');
    errorMsg.classList.add('error-message');
    errorMsg.textContent = `${field.name} is required.`;
    field.parentNode.appendChild(errorMsg);
    
    return false; // Return false if validation fails
  }

  // If the field is valid, remove the error class
  field.classList.remove('error');
  return true; // Return true if validation passes
}



/**
 * Initialize playground event listeners
 */
function initPlaygroundEvents() {
  if (DOM.playgroundForm) {
    // Algorithm selection
    DOM.playgroundForm.querySelector('#playgroundAlgorithm').addEventListener('change', (e) => {
      updatePlaygroundAlgorithm(e.target.value);
    });
    
    // Dataset selection
    DOM.playgroundForm.querySelector('#playgroundDataset').addEventListener('change', (e) => {
      STATE.playground.activeDataset = e.target.value;
      updatePlaygroundVisualization();
    });
    
    // Visualization type
    DOM.playgroundForm.querySelectorAll('input[name="visualizationType"]').forEach((radio) => {
      radio.addEventListener('change', (e) => {
        STATE.playground.visualizationType = e.target.value;
        updatePlaygroundVisualization();
      });
    });
    
    // Train button
    DOM.playgroundForm.querySelector('.train-btn').addEventListener('click', runPlaygroundTraining);
    
    // Hyperparameter changes
    DOM.playgroundForm.querySelector('.hyperparameters').addEventListener('change', (e) => {
      if (e.target.classList.contains('hyperparam-input')) {
        const paramName = e.target.dataset.param;
        let paramValue;
        
        if (e.target.type === 'checkbox') {
          paramValue = e.target.checked;
        } else if (e.target.type === 'number') {
          paramValue = parseFloat(e.target.value);
        } else {
          paramValue = e.target.value;
        }
        
        STATE.playground.hyperparameters[paramName] = paramValue;
        updatePlaygroundVisualization();
      }
    });
  }
}

/**
 * Initialize project event listeners
 */
function initProjectEvents() {
  if (DOM.projectForm) {
    // Project form submission
    DOM.projectForm.addEventListener('submit', handleProjectSubmit);
    
    // Project selection
    document.querySelectorAll('.project-card').forEach((card) => {
      card.addEventListener('click', (e) => {
        const projectId = e.currentTarget.dataset.projectId;
        showProjectDetail(projectId);
      });
    });
    
    // Project completion
    document.querySelectorAll('.complete-project').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        const projectId = e.currentTarget.dataset.projectId;
        completeProject(projectId);
      });
    });
  }
}

/**
 * Initialize scroll and resize event listeners
 */
function initScrollEvents() {
  window.addEventListener('scroll', handleScroll);
  window.addEventListener('resize', debounce(handleResize, 200));
}

/**
 * Initialize keyboard shortcuts
 */
function initKeyboardShortcuts() {
  document.addEventListener('keydown', (e) => {
    // Escape key closes open modals/menus
    if (e.key === 'Escape') {
      if (STATE.isMenuOpen) toggleMenu(false);
      if (STATE.currentAlgorithm) closeAlgorithmDetail();
    }
    
    // Konami code tracking
    trackKonamiCode(e);
    
    // Navigation shortcuts (when not in input)
    if (!['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
      handleNavigationShortcuts(e);
    }
  });
}

/**
 * Initialize UI and theme event listeners
 */
function initUIEvents() {
  // Theme cycling with Ctrl+T
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 't') {
      cycleTheme();
    }
  });
  
  // Dark mode toggle with Ctrl+D
  document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'd') {
      toggleDarkMode();
    }
  });
  
  // Theme selector
  document.querySelectorAll('.theme-selector').forEach((selector) => {
    selector.addEventListener('click', (e) => {
      const theme = e.target.dataset.theme;
      if (theme) {
        applyTheme(theme);
      }
    });
  });
}

// =============================================
// SECTION: Core Functionality
// =============================================

/**
 * Toggle navigation menu state
 * @param {boolean} [forceState] - Optional forced state
 */
function toggleMenu(forceState) {
  STATE.isMenuOpen = forceState !== undefined ? forceState : !STATE.isMenuOpen;
  
  if (STATE.isMenuOpen) {
    DOM.navLinks.classList.add('active');
    DOM.menuToggle.innerHTML = '<i class="fas fa-times"></i>';
    document.body.style.overflow = 'hidden';
  } else {
    DOM.navLinks.classList.remove('active');
    DOM.menuToggle.innerHTML = '<i class="fas fa-bars"></i>';
    document.body.style.overflow = '';
  }
}

/**
 * Handle search functionality
 */
function handleSearch() {
  const term = DOM.searchInput.value.trim().toLowerCase();
  console.log("Search fired:", term);

  if (term.length < 2) {
    // Show all cards
    document.querySelectorAll('.algorithm-card').forEach((card) => {
      card.style.display = 'block';
    });
    return;
  }

  document.querySelectorAll('.algorithm-card').forEach((card) => {
    const title = card.querySelector('.card-title')?.textContent.toLowerCase() || '';
    const descEl = card.querySelector('.card-description p');
    const cardDescription = descEl ? descEl.textContent.toLowerCase() : '';
    const tags = Array.from(card.querySelectorAll('.tag')).map(tag => tag.textContent.toLowerCase());

    const matches = title.includes(term) || 
                    cardDescription.includes(term) || 
                    tags.some(tag => tag.includes(term));

    card.style.display = matches ? 'block' : 'none';
  });
  document.querySelector('#dashboard').scrollIntoView({ behavior: 'smooth' });
}


/**
 * Handle search button click
 */
function handleSearchClick() {
  handleSearch();
  DOM.searchInput.focus();
}

/**
 * Show algorithm detail view
 * @param {string} algorithmId - ID of algorithm to show
 */
/**
 * Show algorithm detail view
 * @param {string} algorithmId - ID of algorithm to show
 */
function showAlgorithmDetail(algorithmId) {
  // Prevent default if coming from click event
  if (typeof event !== 'undefined') {
    event.preventDefault();
    event.stopPropagation();
  }

  STATE.currentAlgorithm = algorithmId;

  // Ensure details container exists
  if (!DOM.algorithmDetailsContainer) {
    DOM.algorithmDetailsContainer = document.getElementById('algorithmDetailsContainer');
    if (!DOM.algorithmDetailsContainer) {
      console.error('Algorithm details container not found');
      return;
    }
  }

  // Generate details if they don't exist
  let detailElement = document.getElementById(`${algorithmId}-detail`);
  if (!detailElement) {
    generateAlgorithmDetails();
    detailElement = document.getElementById(`${algorithmId}-detail`);
  }

  // Hide all detail views
  document.querySelectorAll('.algorithm-detail').forEach((detail) => {
    detail.style.display = 'none';
    detail.classList.remove('active');
  });

  // Show and scroll to selected detail
  if (detailElement) {
    // Store scroll position
    STATE.preDetailScrollPosition = window.scrollY;

    detailElement.style.display = 'block';
    detailElement.classList.add('active');
    document.body.classList.add('detail-view-active');

    // Ensure concept tab is active and visible, others are hidden
    const conceptTab = detailElement.querySelector('#concept-tab');
    const otherTabs = detailElement.querySelectorAll('.tab-pane:not(#concept-tab)');
    
    if (conceptTab) {
      conceptTab.style.display = 'block';
      conceptTab.classList.add('active');
    }
    
    otherTabs.forEach(tab => {
      tab.style.display = 'none';
      tab.classList.remove('active');
    });

    // Set active tab button
    const tabButtons = detailElement.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
      btn.classList.remove('active');
      if (btn.dataset.tab === 'concept') {
        btn.classList.add('active');
      }
    });

    // Smooth scroll to it after render
    setTimeout(() => {
      detailElement.scrollIntoView({
        behavior: 'smooth',
        block: 'start'
      });
      // Ensure the detail is focused for accessibility
      detailElement.focus();
    }, 100);

    // Track user viewing the algorithm
    trackAlgorithmView(algorithmId);
  } else {
    console.error(`Detail element not found for algorithm: ${algorithmId}`);
  }
}

/**
 * Close algorithm detail view
 */
function closeAlgorithmDetail() {
  document.querySelectorAll('.algorithm-detail').forEach((detail) => {
    detail.style.display = 'none';
  });
  
  document.body.style.overflow = '';
  STATE.currentAlgorithm = null;
  
  // Clean up visualizer if available
  if (typeof window.cleanupVisualizer === 'function') {
    window.cleanupVisualizer();
  }

  // Also clean up fullscreen functionality
  if (window.cleanupVisualizers && window.cleanupVisualizers[STATE.currentAlgorithm]) {
    window.cleanupVisualizers[STATE.currentAlgorithm]();
    delete window.cleanupVisualizers[STATE.currentAlgorithm];
  }

}


/**
 * Show code preview for an algorithm
 * @param {string} algorithmId - ID of algorithm
 * @param {HTMLElement} card - Card element to show preview in
 */
function showCodePreview(algorithmId, card) {
  const placeholder = card.querySelector('.visualization-placeholder');
  
  // Show loading state
  placeholder.innerHTML = `
    <div class="code-preview">
      <pre><code class="language-python"># ${algorithmId} implementation\n\n# Loading code preview...</code></pre>
    </div>
  `;
  
  // Load actual code after delay
  setTimeout(() => {
    const algorithm = ALGORITHMS.find(a => a.id === algorithmId);
    const codeElement = placeholder.querySelector('code');
    
    if (algorithm && algorithm.implementations && algorithm.implementations.python) {
      codeElement.textContent = algorithm.implementations.python.code;
    } else {
      codeElement.textContent = '# Code example not available';
    }
    
    // Apply syntax highlighting
    if (window.hljs) {
      hljs.highlightElement(codeElement);
    }
    
    // Add copy button
    addCopyButton(placeholder);
  }, 500);
}

/**
 * Add copy button to code preview
 * @param {HTMLElement} container - Container element
 */
function addCopyButton(container) {
  const copyBtn = document.createElement('button');
  copyBtn.className = 'copy-code-btn';
  copyBtn.innerHTML = '<i class="far fa-copy"></i>';
  copyBtn.title = 'Copy to clipboard';
  
  copyBtn.addEventListener('click', () => {
    const code = container.querySelector('code').textContent;
    navigator.clipboard.writeText(code).then(() => {
      copyBtn.innerHTML = '<i class="fas fa-check"></i>';
      setTimeout(() => {
        copyBtn.innerHTML = '<i class="far fa-copy"></i>';
      }, 2000);
    });
  });
  
  container.appendChild(copyBtn);
}

/**
 * Handle tab switching in algorithm details
 * @param {Event} e - Click event
 */
/**
 * Handle tab switching in algorithm details
 * @param {Event} e - Click event
 */
function handleTabSwitch(e) {
  const tabName = e.target.dataset.tab;
  const tabContainer = e.target.closest('.content-tabs');
  
  // Update active tab button
  tabContainer.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.classList.remove('active');
  });
  e.target.classList.add('active');
  
  // Update active tab content - hide all, then show the selected one
  tabContainer.querySelectorAll('.tab-pane').forEach((pane) => {
    pane.classList.remove('active');
    pane.style.display = 'none';
  });
  
  const activePane = tabContainer.querySelector(`#${tabName}-tab`);
  if (activePane) {
    activePane.style.display = 'block';
    activePane.classList.add('active');
  }
  
  // Update state
  STATE.currentTab = tabName;
  
  // Initialize visualizations if needed
  if (tabName === 'visualization' && typeof window.initVisualizer === 'function') {
    // Small delay to ensure the tab is visible before initializing
    setTimeout(() => {
      window.initVisualizer(STATE.currentAlgorithm);
    }, 100);
  }
  
  // Apply syntax highlighting for code tab
  if (tabName === 'code' && window.hljs) {
    setTimeout(() => {
      const codeBlocks = activePane.querySelectorAll('pre code');
      codeBlocks.forEach((block) => {
        hljs.highlightElement(block);
      });
    }, 100);
  }
}

/**
 * Handle code language tab switching
 * @param {Event} e - Click event
 */
function handleCodeTabSwitch(e) {
  const lang = e.target.dataset.lang;
  const tabContainer = e.target.closest('.code-tabs');
  
  // Update active tab button
  tabContainer.querySelectorAll('.code-tab-btn').forEach((btn) => {
    btn.classList.remove('active');
  });
  e.target.classList.add('active');
  
  // Update active code content
  tabContainer.closest('.code-container').querySelectorAll('.code-content pre').forEach((pre) => {
    pre.classList.remove('active');
    if (pre.dataset.lang === lang) {
      pre.classList.add('active');
    }
  });
  
  // Update state
  STATE.currentCodeLang = lang;
}

/**
 * Navigate to next/previous algorithm
 * @param {Event} e - Click event
 */
function navigateAlgorithm(e) {
  const currentIndex = ALGORITHMS.findIndex(a => a.id === STATE.currentAlgorithm);
  let newIndex;
  
  if (e.currentTarget.classList.contains('prev-algorithm')) {
    newIndex = (currentIndex - 1 + ALGORITHMS.length) % ALGORITHMS.length;
  } else {
    newIndex = (currentIndex + 1) % ALGORITHMS.length;
  }
  
  showAlgorithmDetail(ALGORITHMS[newIndex].id);
}

// =============================================
// SECTION: Algorithm Data Generation
// =============================================

/**
 * Generate algorithm cards and add to DOM
 */
function generateAlgorithmCards() {
  if (!DOM.algorithmGrid) return;
  
  // Clear existing cards (if any)
  DOM.algorithmGrid.innerHTML = '';
  
  // Create cards for each algorithm
  ALGORITHMS.forEach((algorithm) => {
    const card = document.createElement('div');
    card.className = 'algorithm-card';
    card.dataset.category = algorithm.category;
    card.dataset.difficulty = algorithm.difficulty;
    card.dataset.tags = algorithm.tags.join(',').toLowerCase();
    
    card.innerHTML = `
      <div class="card-header">
        <div class="card-badge ${algorithm.difficulty}">
          ${CONFIG.difficulties[algorithm.difficulty].name}
        </div>
        <h3 class="card-title">${algorithm.title}</h3>
        <div class="card-tags">
          ${algorithm.tags.map((tag) => `<span class="tag">${tag}</span>`).join('')}
        </div>
      </div>
      
      <div class="card-visualization">
        <div class="visualization-placeholder" id="${algorithm.id}-vis">
          <div class="placeholder-content">
            <i class="fas fa-${algorithm.icon}"></i>
            <span>Interactive visualization loading...</span>
          </div>
        </div>
      </div>
      
      <div class="card-description">
        <p>${algorithm.description}</p>
      </div>
      
      <div class="card-actions">
        <button class="action-btn explore-btn" data-algorithm="${algorithm.id}">
          <i class="fas fa-play"></i> Explore
        </button>
        <button class="action-btn code-btn" data-algorithm="${algorithm.id}">
          <i class="fas fa-code"></i> Code
        </button>
      </div>
    `;
    
    DOM.algorithmGrid.appendChild(card);
  });
  
  // Update DOM references
  DOM.algorithmCards = document.querySelectorAll('.algorithm-card');
}

/**
 * Generate algorithm detail sections
 */
/**
 * Generate algorithm detail sections
 */
function generateAlgorithmDetails() {
  if (!DOM.algorithmDetailsContainer) return;
  
  // Clear existing details (if any)
  DOM.algorithmDetailsContainer.innerHTML = '';
  
  // Create detail sections for each algorithm
  ALGORITHMS.forEach((algorithm) => {
    const detailSection = document.createElement('div');
    detailSection.className = 'algorithm-detail';
    detailSection.id = `${algorithm.id}-detail`;
    
    // Build the detail section HTML
    detailSection.innerHTML = generateAlgorithmDetailHTML(algorithm);
    
    DOM.algorithmDetailsContainer.appendChild(detailSection);
  });
  
  // After generating, ensure only concept tab is visible in each detail section
  document.querySelectorAll('.algorithm-detail').forEach(detail => {
    const conceptTab = detail.querySelector('#concept-tab');
    const otherTabs = detail.querySelectorAll('.tab-pane:not(#concept-tab)');
    
    if (conceptTab) {
      conceptTab.classList.add('active');
    }
    
    otherTabs.forEach(tab => {
      tab.classList.remove('active');
    });
  });
}

/**
 * Generate HTML for algorithm detail section
 * @param {Object} algorithm - Algorithm data
 * @return {string} - Generated HTML
 */
/**
 * Generate HTML for algorithm detail section
 * @param {Object} algorithm - Algorithm data
 * @return {string} - Generated HTML
 */
function generateAlgorithmDetailHTML(algorithm) {
  return `
    <div class="detail-header">
      <div class="header-content">
        <h3 class="algorithm-name">${algorithm.title}</h3>
        <div class="algorithm-meta">
          <span class="badge ${algorithm.difficulty}">
            ${CONFIG.difficulties[algorithm.difficulty].name}
          </span>
          <span class="algorithm-type">
            ${CONFIG.categories[algorithm.category]}
          </span>
          <span class="algorithm-popularity">
            <i class="fas fa-star"></i> ${Math.round(algorithm.popularity * 100)}%
          </span>
        </div>
      </div>
      <div class="header-actions">
        <button class="close-detail">
          <i class="fas fa-times"></i>
        </button>
      </div>
    </div>
    
    <div class="detail-content">
      <div class="content-tabs">
        <div class="tabs-header">
          <button class="tab-btn active" data-tab="concept">
            <i class="fas fa-lightbulb"></i> Concept
          </button>
          <button class="tab-btn" data-tab="code">
            <i class="fas fa-code"></i> Implementation
          </button>
          <button class="tab-btn" data-tab="visualization">
            <i class="fas fa-eye"></i> Visualization
          </button>
          <button class="tab-btn" data-tab="proscons">
            <i class="fas fa-balance-scale"></i> Pros & Cons
          </button>
          <button class="tab-btn" data-tab="uses">
            <i class="fas fa-rocket"></i> Use Cases
          </button>
          <button class="tab-btn" data-tab="quiz">
            <i class="fas fa-question-circle"></i> Quiz
          </button>
          <button class="tab-btn" data-tab="projects">
            <i class="fas fa-tasks"></i> Projects
          </button>
        </div>
        
        <div class="tabs-content">
          <!-- Generate ALL tabs but only show concept initially -->
          ${generateConceptTab(algorithm)}
          ${generateCodeTab(algorithm)}
          ${generateVisualizationTab(algorithm)}
          ${generateProsConsTab(algorithm)}
          ${generateUsesTab(algorithm)}
          ${generateQuizTab(algorithm)}
          ${generateProjectsTab(algorithm)}
        </div>
      </div>
    </div>
    
    <div class="detail-footer">
      <div class="footer-actions">
        <button class="btn-secondary prev-algorithm">
          <i class="fas fa-arrow-left"></i> Previous
        </button>
        <button class="btn-secondary next-algorithm">
          Next <i class="fas fa-arrow-right"></i>
        </button>
      </div>
      <div class="progress-tracker">
        <div class="progress-text">Progress: <span>0%</span></div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: 0%"></div>
        </div>
      </div>
    </div>
  `;
}


/**
 * Generate concept tab content
 * @param {Object} algorithm - Algorithm data
 * @return {string} - Generated HTML
 */
function generateConceptTab(algorithm) {
  return `
    <div class="tab-pane active" id="concept-tab">
      <div class="concept-container">
        <div class="concept-intro">
          <h4>Core Concept</h4>
          <p>${algorithm.concept.overview}</p>
          
          <div class="concept-analogy">
            <h5>Real-world Analogy</h5>
            <p>${algorithm.concept.analogy}</p>
          </div>
          
          <div class="concept-history">
            <h5>History & Development</h5>
            <p>${algorithm.concept.history}</p>
          </div>
        </div>
        
        ${algorithm.concept.mathematicalFormulation ? `
        <div class="concept-formula">
          <h4>Mathematical Foundation</h4>
          <div class="formula-box">
            ${algorithm.concept.mathematicalFormulation.equation ? `
            <div class="formula-display">
              ${algorithm.concept.mathematicalFormulation.equation}
            </div>
            ` : ''}
            <div class="formula-legend">
              <ul>
                ${algorithm.concept.mathematicalFormulation.variables ? algorithm.concept.mathematicalFormulation.variables.map(varItem => `
                  <li><strong>${varItem.symbol}</strong>: ${varItem.description}</li>
                `).join('') : ''}
              </ul>

            </div>
          </div>
          
          ${algorithm.concept.mathematicalFormulation.costFunction ? `
          <div class="formula-details">
            <h5>Cost Function</h5>
            <p>${algorithm.concept.mathematicalFormulation.costFunction}</p>
          </div>
          ` : ''}
          
          ${algorithm.concept.mathematicalFormulation.optimization ? `
          <div class="formula-details">
            <h5>Optimization</h5>
            <p>${algorithm.concept.mathematicalFormulation.optimization}</p>
          </div>
          ` : ''}
        </div>
        ` : ''}
        
        ${algorithm.concept.assumptions ? `
        <div class="concept-assumptions">
          <h4>Key Assumptions</h4>
          <ul>
            ${algorithm.concept.assumptions.map(assumption => `
            <li>${assumption}</li>
            `).join('')}
          </ul>
        </div>
        ` : ''}
      </div>
    </div>
  `;
}

/**
 * Generates comprehensive code tab content for algorithm display, including
 * multi-language implementations, syntax highlighting, library dependencies,
 * and interactive elements.
 * 
 * @param {Object} algorithm - Algorithm data object containing implementations
 * @returns {string} - HTML string for the code tab component
 */
function generateCodeTab(algorithm) {
  // Validate input
  if (!algorithm || !algorithm.implementations) {
    console.error('Invalid algorithm data provided to generateCodeTab');
    return '<div class="error">Error loading code implementations</div>';
  }

  // Supported languages with their icons and display names
  const languageMetadata = {
    python: { icon: 'fab fa-python', name: 'Python' },
    r: { icon: 'fab fa-r-project', name: 'R' },
    javascript: { icon: 'fab fa-js-square', name: 'JavaScript' },
    cpp: { icon: 'fas fa-file-code', name: 'C++' },
    java: { icon: 'fab fa-java', name: 'Java' }
  };

  // Generate language tabs for available implementations
  const generateLanguageTabs = () => {
    return Object.entries(languageMetadata)
      .filter(([lang]) => algorithm.implementations[lang]?.code)
      .map(([lang, meta], index) => `
        <button class="code-tab-btn ${index === 0 ? 'active' : ''}" 
                data-lang="${lang}"
                aria-label="Show ${meta.name} implementation">
          <i class="${meta.icon}"></i> ${meta.name}
          ${algorithm.implementations[lang].version ? `
            <span class="version-badge" title="${meta.name} version">v${algorithm.implementations[lang].version}</span>
          ` : ''}
        </button>
      `).join('');
  };

  // Generate code blocks for available implementations
  const generateCodeBlocks = () => {
    return Object.entries(languageMetadata)
      .filter(([lang]) => algorithm.implementations[lang]?.code)
      .map(([lang], index) => `
        <pre class="code-block ${index === 0 ? 'active' : ''}" 
             data-lang="${lang}"
             aria-labelledby="code-tab-${lang}">
          <code class="language-${lang}">${escapeHtml(algorithm.implementations[lang].code)}</code>
          ${generateCodeFooter(algorithm.implementations[lang])}
        </pre>
      `).join('');
  };

  // Generate footer for each code block (complexity, author, etc.)
  const generateCodeFooter = (impl) => {
    return `
      <div class="code-footer">
        ${impl.timeComplexity ? `
          <div class="complexity-info">
            <span class="complexity-label">Time:</span>
            <span class="complexity-value">${impl.timeComplexity}</span>
            ${impl.spaceComplexity ? `
              <span class="complexity-separator">|</span>
              <span class="complexity-label">Space:</span>
              <span class="complexity-value">${impl.spaceComplexity}</span>
            ` : ''}
          </div>
        ` : ''}
        
        ${impl.author ? `
          <div class="code-author">
            <i class="fas fa-user-edit"></i>
            <span class="author-name">${impl.author.name}</span>
            ${impl.author.link ? `
              <a href="${impl.author.link}" target="_blank" rel="noopener">
                <i class="fas fa-external-link-alt"></i>
              </a>
            ` : ''}
          </div>
        ` : ''}
        
        ${impl.lastUpdated ? `
          <div class="code-updated">
            <i class="far fa-calendar-alt"></i>
            Last updated: ${new Date(impl.lastUpdated).toLocaleDateString()}
          </div>
        ` : ''}
      </div>
    `;
  };

  // Generate library badges with optional documentation links
  const generateLibraryBadges = () => {
    const libs = new Set();
    
    // Collect all libraries from all implementations
    Object.values(algorithm.implementations).forEach(impl => {
      if (impl.libraries) {
        impl.libraries.forEach(lib => libs.add(lib));
      }
    });

    if (libs.size === 0) return '';
    
    return `
      <div class="code-libraries">
        <h4><i class="fas fa-book"></i> Libraries Used</h4>
        <div class="library-badges">
          ${Array.from(libs).map(lib => {
            const [name, version] = lib.split('@');
            return `
              <a href="https://pypi.org/project/${name}/" target="_blank" rel="noopener" class="library-badge">
                <i class="fas fa-cube"></i>
                <span class="library-name">${name}</span>
                ${version ? `<span class="library-version">${version}</span>` : ''}
              </a>
            `;
          }).join('')}
        </div>
      </div>
    `;
  };

  // Generate implementation notes if available
  const generateImplementationNotes = () => {
    const notes = [];
    
    Object.entries(algorithm.implementations).forEach(([lang, impl]) => {
      if (impl.notes) {
        notes.push(`
          <div class="implementation-note" data-lang="${lang}">
            <h5><i class="${languageMetadata[lang]?.icon || 'fas fa-code'}"></i> ${languageMetadata[lang]?.name || lang} Notes</h5>
            <p>${impl.notes}</p>
          </div>
        `);
      }
    });

    return notes.length > 0 ? `
      <div class="implementation-notes">
        <h4><i class="fas fa-sticky-note"></i> Implementation Notes</h4>
        ${notes.join('')}
      </div>
    ` : '';
  };

  // Helper function to escape HTML for code display
  const escapeHtml = (unsafe) => {
    return unsafe.replace(/[&<>'"]/g, match => ({
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    }[match]));
  };

  return `
    <div class="tab-pane code-tab-container" id="code-tab" role="tabpanel" aria-labelled-by="code-tab">
      <div class="code-container">
        <div class="code-header">
          <h3 class="code-title">
            <i class="fas fa-code"></i> Implementation Details
          </h3>
          <div class="code-actions">
            <button class="btn btn-sm btn-outline-secondary copy-code" title="Copy code to clipboard">
              <i class="far fa-copy"></i> Copy
            </button>
            <button class="btn btn-sm btn-outline-secondary expand-code" title="Expand code view">
              <i class="fas fa-expand"></i>
            </button>
          </div>
        </div>
        
        <div class="code-tabs-wrapper">
          <div class="code-tabs" role="tablist">
            ${generateLanguageTabs()}
          </div>
          
          <div class="code-content" role="tabpanel">
            ${generateCodeBlocks()}
          </div>
        </div>
        
        ${generateLibraryBadges()}
        ${generateImplementationNotes()}
        
        <div class="code-footer-global">
          <div class="code-license">
            <i class="fas fa-balance-scale"></i>
            <span>License: ${algorithm.license || 'MIT'}</span>
          </div>
          <div class="code-contribute">
            <a href="${algorithm.repoUrl || '#'}" target="_blank" rel="noopener">
              <i class="fas fa-hands-helping"></i> Contribute improvements
            </a>
          </div>
        </div>
      </div>
      
      <div class="code-modal-overlay">
        <div class="code-modal">
          <div class="modal-header">
            <h4>Full Screen Code View</h4>
            <button class="modal-close">&times;</button>
          </div>
          <div class="modal-content"></div>
        </div>
      </div>
    </div>
  `;
}

// Robust expand/zoom for code tabs (works with dynamically injected tabs)
(() => {
  const SELECTORS = {
    container: '.code-container',
    expandBtn: '.expand-code',
    activeBlock: '.code-block.active, .code-pane.active, pre.code-block.active, pre.code-pane.active',
    anyBlock: '.code-block, .code-pane, pre.code-block, pre.code-pane',
    overlay: '.code-modal-overlay',
    modal: '.code-modal',
    modalContent: '.modal-content',
    modalClose: '.modal-close'
  };

  function findContainer(el) {
    return el.closest(SELECTORS.container);
  }

  function getActiveBlock(container) {
    return (
      container.querySelector(SELECTORS.activeBlock) ||
      container.querySelector(SELECTORS.anyBlock)
    );
  }

  function ensureOverlay(container) {
    let overlay = container.querySelector(SELECTORS.overlay);
    if (!overlay) {
      // Safety: create modal if your template wasn't included for some reason
      overlay = document.createElement('div');
      overlay.className = 'code-modal-overlay';
      overlay.innerHTML = `
        <div class="code-modal">
          <div class="modal-header">
            <h4>Full Screen Code View</h4>
            <button class="modal-close" aria-label="Close">&times;</button>
          </div>
          <div class="modal-content"></div>
        </div>
      `;
      container.appendChild(overlay);
    }
    return overlay;
  }

  function openModal(container) {
    const overlay = ensureOverlay(container);
    const content = overlay.querySelector(SELECTORS.modalContent);
    const active = getActiveBlock(container);

    if (!active) {
      console.warn('[code expand] No code block found to expand.');
      return;
    }

    // Clone the active block so we don't move it out of the tab
    const clone = active.cloneNode(true);
    // Make sure it's visible inside the modal regardless of tab state
    clone.classList.add('active');

    // Insert & (optionally) re-highlight
    content.innerHTML = '';
    content.appendChild(clone);

    // If Prism is available, re-highlight inside modal
    if (window.Prism) {
      content.querySelectorAll('code').forEach((el) => {
        try { Prism.highlightElement(el); } catch {}
      });
    }

    overlay.classList.add('active');
    document.body.style.overflow = 'hidden'; // prevent background scroll
  }

  function closeModal(overlay) {
    overlay.classList.remove('active');
    const content = overlay.querySelector(SELECTORS.modalContent);
    if (content) content.innerHTML = '';
    document.body.style.overflow = '';
  }

  // Event delegation for clicks
  document.addEventListener('click', (e) => {
    // Open
    const expandBtn = e.target.closest(SELECTORS.expandBtn);
    if (expandBtn) {
      e.preventDefault();
      const container = findContainer(expandBtn);
      if (container) openModal(container);
      return;
    }

    // Close on ✖ button
    const closeBtn = e.target.closest(SELECTORS.modalClose);
    if (closeBtn) {
      const overlay = closeBtn.closest(SELECTORS.overlay);
      if (overlay) closeModal(overlay);
      return;
    }

    // Close on overlay click (but not when clicking inside the modal)
    const overlay = e.target.classList?.contains('code-modal-overlay') ? e.target : null;
    if (overlay) closeModal(overlay);
  });

  // Close on ESC
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      document.querySelectorAll(`${SELECTORS.overlay}.active`).forEach(closeModal);
    }
  });
})();


// Clipboard copy logic
document.addEventListener('click', function (e) {
  if (e.target.closest('.copy-code')) {
    const codeContainer = e.target.closest('.code-tab-container');
    if (!codeContainer) return;

    // Find the active code block inside the same container
    const activeCodeBlock = codeContainer.querySelector('.code-block.active code');
    if (!activeCodeBlock) return;

    // Get code text
    const codeText = activeCodeBlock.innerText;

    // Copy to clipboard
    navigator.clipboard.writeText(codeText).then(() => {
      // Feedback to user
      const btn = e.target.closest('.copy-code');
      const original = btn.innerHTML;
      btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
      setTimeout(() => (btn.innerHTML = original), 2000);
    }).catch(err => {
      console.error('Failed to copy code: ', err);
    });
  }
});


/**
 * Generate visualization tab content
 * @param {Object} algorithm - Algorithm data
 * @return {string} - Generated HTML
 */
// ✅ Safe version with fallback if visualization or parameters are missing
// Updated functions for main.js

function generateVisualizationTab(algorithm) {
  const visualization = algorithm.visualization || {};

  const defaultParams = {
    interactive: true,
    show_grid: true,
    show_axes: true,
    animation_duration: 1500
  };

  const visualizationParams = {
    ...defaultParams,
    ...(visualization.parameters || {})
  };

  return `
    <div class="tab-pane" id="visualization-tab">
      <div class="visualization-container">

        <!-- Dynamic controls -->
        <div class="visualization-controls-dynamic" id="${algorithm.id}-controls">
          <div class="controls-loading">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Loading interactive controls...</p>
          </div>
        </div>

        <!-- Visualization canvas -->
        <div class="visualization-canvas" id="${algorithm.id}-visualization">
          <div class="visualization-loading">
            <i class="fas fa-spinner fa-spin"></i>
            <p>Loading interactive visualization...</p>
          </div>
        </div>

        <!-- Action buttons -->
        <div class="visualization-actions">
          <div class="action-buttons">
            <button class="btn-secondary reset-visualization" data-algorithm="${algorithm.id}">
              <i class="fas fa-redo"></i> Reset
            </button>
            <button class="btn-primary animate-visualization" data-algorithm="${algorithm.id}">
              <i class="fas fa-play"></i> Restart Animation
            </button>

            ${visualization.types && visualization.types.length > 1 ? `
              <select class="visualization-type-selector" data-algorithm="${algorithm.id}">
                ${visualization.types.map(type => `
                  <option value="${type.value}" ${type.default ? 'selected' : ''}>${type.label}</option>
                `).join('')}
              </select>
            ` : ''}

            <div class="visualization-toggles">
              <button class="btn-toggle ${visualizationParams.show_grid ? 'active' : ''}" 
                data-algorithm="${algorithm.id}" data-param="show_grid" title="Toggle Grid">
                <i class="fas fa-border-style"></i>
              </button>
              <button class="btn-toggle ${visualizationParams.show_axes ? 'active' : ''}" 
                data-algorithm="${algorithm.id}" data-param="show_axes" title="Toggle Axes">
                <i class="fas fa-crosshairs"></i>
              </button>
            </div>
          </div>
        </div>

        <!-- Visualization info -->
        <div class="visualization-info">
          <div class="visualization-description">
            <h4>${algorithm.title} Visualization</h4>
            <p>${visualization.description || `Interact with the visualization to understand how ${algorithm.title} works.`}</p>
          </div>

          ${visualization.instructions ? `
            <div class="visualization-instructions">
              <h5><i class="fas fa-lightbulb"></i> How to Use:</h5>
              <ul>
                ${visualization.instructions.map(ins => `<li>${ins}</li>`).join('')}
              </ul>
            </div>
          ` : ''}

          ${visualization.performanceTips ? `
            <div class="visualization-tips">
              <h5><i class="fas fa-info-circle"></i> Performance Tips:</h5>
              <ul>
                ${visualization.performanceTips.map(tip => `<li>${tip}</li>`).join('')}
              </ul>
            </div>
          ` : ''}

          <div class="visualization-parameters">
            <h5><i class="fas fa-sliders-h"></i> Current Parameters:</h5>
            <div id="${algorithm.id}-params-display"></div>
          </div>
        </div>

      </div>
    </div>
  `;
}


/**
 * Initialize visualizer
 */
window.initVisualizer = function(algorithmId) {
  if (!algorithmId) {
    console.warn('initVisualizer called without algorithm ID');
    return;
  }
  
  const algorithm = ALGORITHMS?.find(a => a.id === algorithmId);
  if (!algorithm || !algorithm.visualization) {
    console.warn('No visualization configuration found for algorithm:', algorithmId);
    showVisualizationError(`${algorithmId}-visualization`, 'No visualization available for this algorithm');
    return;
  }
  
  const visualization = algorithm.visualization;
  const containerId = `${algorithmId}-visualization`;
  const controlsContainerId = `${algorithmId}-controls`;
  
  // Remove loading indicator
  const loadingElement = document.querySelector(`#${containerId} .visualization-loading`);
  if (loadingElement) {
    loadingElement.style.display = 'none';
  }
  
  // Get visualization configuration
  const visualizationType = visualization.defaultType || 'default';
  const params = {
    ...visualization.parameters,
    interactive: true,
    controlsContainer: controlsContainerId
  };
  
  // Get the visualizer function using the visualizerKey
  const visualizerFn = window.visualizers[visualization.visualizerKey];
  if (visualizerFn) {
    try {
      visualizerFn(containerId, visualizationType, params);
      setupVisualizationEventHandlers(algorithmId, algorithm, visualizerFn, containerId, params);
    } catch (error) {
      console.error('Error initializing visualizer:', error);
      showVisualizationError(containerId, error.message);
    }
  } else {
    console.warn('No visualizer function found for key:', visualization.visualizerKey);
    showVisualizationError(containerId, `Visualizer "${visualization.visualizerKey}" not found`);
  }
};

/**
 * Setup visualization event handlers
 */
function setupVisualizationEventHandlers(algorithmId, algorithm, visualizerFn, containerId, baseParams) {
  const visualization = algorithm.visualization;
  
  // Reset button handler
  const resetBtn = document.querySelector(`.reset-visualization[data-algorithm="${algorithmId}"]`);
  if (resetBtn) {
    resetBtn.addEventListener('click', () => {
      // Clear the canvas and reinitialize
      const canvas = document.querySelector(`#${containerId} canvas`);
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
      
      // Clear controls container
      const controlsContainer = document.getElementById(`${algorithmId}-controls`);
      if (controlsContainer) {
        controlsContainer.innerHTML = '';
      }
      
      // Reinitialize with default parameters
      const defaultParams = {
        ...visualization.parameters,
        interactive: true,
        controlsContainer: `${algorithmId}-controls`
      };
      
      visualizerFn(containerId, visualization.defaultType, defaultParams);
    });
  }
  
  // Animate button handler
  const animateBtn = document.querySelector(`.animate-visualization[data-algorithm="${algorithmId}"]`);
  if (animateBtn) {
    animateBtn.addEventListener('click', () => {
      // Get current parameters from controls if they exist
      const currentParams = getCurrentVisualizationParams(algorithmId, baseParams);
      visualizerFn(containerId, getCurrentVisualizationType(algorithmId, visualization), currentParams);
    });
  }
  
  // Visualization type selector handler
  const typeSelector = document.querySelector(`.visualization-type-selector[data-algorithm="${algorithmId}"]`);
  if (typeSelector) {
    typeSelector.addEventListener('change', (e) => {
      const newType = e.target.value;
      const currentParams = getCurrentVisualizationParams(algorithmId, baseParams);
      visualizerFn(containerId, newType, currentParams);
    });
  }
}

/**
 * Helper function to get current visualization parameters
 */
function getCurrentVisualizationParams(algorithmId, baseParams) {
  // This would extract current values from the dynamic controls
  // For now, return the base params
  return {
    ...baseParams,
    forceRestart: true
  };
}

/**
 * Helper function to get current visualization type
 */
function getCurrentVisualizationType(algorithmId, visualization) {
  const typeSelector = document.querySelector(`.visualization-type-selector[data-algorithm="${algorithmId}"]`);
  return typeSelector ? typeSelector.value : visualization.defaultType;
}

/**
 * Show visualization error
 */
function showVisualizationError(containerId, message) {
  const container = document.getElementById(containerId);
  if (container) {
    container.innerHTML = `
      <div class="visualization-error">
        <div class="error-icon">
          <i class="fas fa-exclamation-triangle"></i>
        </div>
        <div class="error-content">
          <h4>Visualization Unavailable</h4>
          <p>${message}</p>
          <small>Please check the console for technical details.</small>
        </div>
      </div>
    `;
  }
}

/**
 * Initialize visualization
 */
function initializeVisualization(algorithmId) {
  // Wait for DOM to be ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      setTimeout(() => initVisualizer(algorithmId), 100);
    });
  } else {
    setTimeout(() => initVisualizer(algorithmId), 100);
  }
}


/**
 * Generate pros & cons tab content
 * @param {Object} algorithm - Algorithm data
 * @return {string} - Generated HTML
 */
function generateProsConsTab(algorithm) {
  return `
    <div class="tab-pane" id="proscons-tab">
      <div class="proscons-container">
        <div class="pros-box">
          <h4><i class="fas fa-check-circle"></i> Strengths</h4>
          <ul>
            ${algorithm.prosCons.strengths.map(strength => `
            <li>${strength}</li>
            `).join('')}
          </ul>
        </div>
        
        <div class="cons-box">
          <h4><i class="fas fa-times-circle"></i> Weaknesses</h4>
          <ul>
            ${algorithm.prosCons.weaknesses.map(weakness => `
            <li>${weakness}</li>
            `).join('')}
          </ul>
        </div>
        
        ${algorithm.comparisons ? `
        <div class="comparison-box">
          <h4><i class="fas fa-not-equal"></i> Comparisons</h4>
          <div class="comparison-table">
            <table>
              <thead>
                <tr>
                  <th>Algorithm</th>
                  <th>Comparison</th>
                </tr>
              </thead>
              <tbody>
                ${algorithm.comparisons.map(comp => `
                <tr>
                  <td>${comp.algorithm}</td>
                  <td>${comp.comparison}</td>
                </tr>
                `).join('')}
              </tbody>
            </table>
          </div>
        </div>
        ` : ''}
      </div>
    </div>
  `;
}

/**
 * Generate use cases tab content
 * @param {Object} algorithm - Algorithm data
 * @return {string} - Generated HTML
 */
function generateUsesTab(algorithm) {
  const useCases = algorithm.useCases || [];

  return `
    <div class="tab-pane" id="uses-tab">
      <div class="uses-container">
        <h4 class="uses-title">Real-world Applications</h4>

        ${useCases.length > 0 ? useCases.map(useCase => `
          <article class="use-case-card">
            <div class="use-case-icon">
              <i class="fas fa-rocket"></i>
            </div>
            <div class="use-case-content">
              <h5 class="use-case-title">${useCase.title}</h5>
              <p class="use-case-description">${useCase.description}</p>
              ${useCase.dataset ? `
              <div class="use-case-dataset">
                <i class="fas fa-database"></i> 
                Dataset: 
                <a href="${useCase.datasetLink || '#'}" target="_blank" rel="noopener noreferrer">
                  ${useCase.dataset}
                </a>
              </div>
              ` : ''}
            </div>
          </article>
        `).join('') : `
          <p class="no-use-cases">No use cases available for this algorithm.</p>
        `}

        <div class="use-case-actions">
          <button class="btn-secondary">
            <i class="fas fa-book"></i> Read Case Studies
          </button>
          <button class="btn-primary">
            <i class="fas fa-play"></i> Watch Applications
          </button>
        </div>
      </div>
    </div>
  `;
}


/**
 * Generate quiz tab content
 * @param {Object} algorithm - Algorithm data
 * @return {string} - Generated HTML
 */
function generateQuizTab(algorithm) {
  return `
    <div class="tab-pane" id="quiz-tab">
      <div class="quiz-container">
        <div class="quiz-header">
          <h4>Test Your Understanding</h4>
          <p>Complete this quiz to earn XP and track your progress</p>
        </div>
        
        <div class="quiz-questions">
          ${algorithm.quiz && Array.isArray(algorithm.quiz) ? algorithm.quiz.map((question, index) => `
            <div class="question-card">
              <div class="question-text">
                ${index + 1}. ${question.question}
              </div>
              <div class="question-options">
                ${question.options && Array.isArray(question.options) ? question.options.map((option, optIndex) => `
                  <label class="option">
                    <input type="radio" name="q${index}" value="${optIndex}" 
                    data-correct="${question.correct === optIndex}">
                    <span class="option-text">${option}</span>
                  </label>
                `).join('') : ''}
              </div>
            </div>
          `).join('') : ''}
        </div>
        
        <div class="quiz-actions">
          <button class="btn-secondary reset-quiz">
            <i class="fas fa-redo"></i> Reset Quiz
          </button>
          <button class="btn-primary submit-quiz">
            <i class="fas fa-check"></i> Submit Quiz
          </button>
        </div>
      </div>
    </div>
  `;
}

/**
 * Generate projects tab content
 * @param {Object} algorithm - Algorithm data
 * @return {string} - Generated HTML
 */
function generateProjectsTab(algorithm) {
  return `
    <div class="tab-pane" id="projects-tab">
      <div class="projects-container">
        <div class="projects-header">
          <h4>Hands-on Projects</h4>
          <p>Apply what you've learned with these guided projects</p>
        </div>
        
        <div class="projects-grid">
          ${algorithm.projects.map((project, index) => `
          <div class="project-card" data-project-id="${algorithm.id}-project-${index}">
            <div class="project-badge ${project.difficulty}">
              ${CONFIG.difficulties[project.difficulty].name}
            </div>
            <div class="project-content">
              <h5>${project.title}</h5>
              <p>${project.description}</p>
              <div class="project-xp">
                <i class="fas fa-star"></i> ${project.xp} XP
              </div>
            </div>
          </div>
          `).join('')}
        </div>
      </div>
    </div>
  `;
}

// =============================================
// SECTION: Playground Functionality
// =============================================

/**
 * Initialize the algorithm playground
 */
function initPlayground() {
  if (!DOM.playgroundSection) return;
  
  // Populate algorithm dropdown
  const algorithmSelect = DOM.playgroundForm.querySelector('#playgroundAlgorithm');
  ALGORITHMS.forEach(algorithm => {
    const option = document.createElement('option');
    option.value = algorithm.id;
    option.textContent = algorithm.title;
    algorithmSelect.appendChild(option);
  });
  
  // Set initial state
  STATE.playground.activeAlgorithm = ALGORITHMS[0].id;
  STATE.playground.hyperparameters = {...CONFIG.defaultHyperparams[ALGORITHMS[0].id]};
  
  // Initialize playground
  updatePlaygroundAlgorithm(ALGORITHMS[0].id);
}

/**
 * Update playground when algorithm changes
 * @param {string} algorithmId - Selected algorithm ID
 */
function updatePlaygroundAlgorithm(algorithmId) {
  const algorithm = ALGORITHMS.find(a => a.id === algorithmId);
  if (!algorithm) return;
  
  STATE.playground.activeAlgorithm = algorithmId;
  STATE.playground.hyperparameters = {...CONFIG.defaultHyperparams[algorithmId]};
  
  // Update dataset options based on algorithm type
  const datasetSelect = DOM.playgroundForm.querySelector('#playgroundDataset');
  datasetSelect.innerHTML = '';
  
  let datasets = [];
  if (algorithm.category === 'supervised' && algorithm.tags.includes('Classification')) {
    datasets = CONFIG.playgroundDatasets.classification;
  } else if (algorithm.category === 'supervised' && algorithm.tags.includes('Regression')) {
    datasets = CONFIG.playgroundDatasets.regression;
  } else if (algorithm.category === 'unsupervised') {
    datasets = CONFIG.playgroundDatasets.clustering;
  }
  
  datasets.forEach(dataset => {
    const option = document.createElement('option');
    option.value = dataset.toLowerCase().replace(' ', '-');
    option.textContent = dataset;
    datasetSelect.appendChild(option);
  });
  
  // Update hyperparameters
  updatePlaygroundHyperparameters(algorithm);
  
  // Update visualization
  updatePlaygroundVisualization();
}

/**
 * Update playground hyperparameter controls
 * @param {Object} algorithm - Algorithm data
 */
function updatePlaygroundHyperparameters(algorithm) {
  const hyperparamsContainer = DOM.playgroundForm.querySelector('.hyperparameters');
  hyperparamsContainer.innerHTML = '<h4>Hyperparameters</h4>';
  
  algorithm.hyperparameters.forEach(param => {
    const paramGroup = document.createElement('div');
    paramGroup.className = 'hyperparam-group';
    
    const label = document.createElement('label');
    label.textContent = param.name;
    label.title = param.description;
    
    let input;
    if (param.type === 'boolean') {
      input = document.createElement('input');
      input.type = 'checkbox';
      input.checked = param.default;
      input.className = 'hyperparam-input';
      input.dataset.param = param.name;
    } else if (param.type === 'range') {
      input = document.createElement('input');
      input.type = 'range';
      input.min = param.min;
      input.max = param.max;
      input.step = param.step || 1;
      input.value = param.default;
      input.className = 'hyperparam-input';
      input.dataset.param = param.name;
      
      // Add value display
      const valueSpan = document.createElement('span');
      valueSpan.className = 'hyperparam-value';
      valueSpan.textContent = param.default;
      
      // Update value display when slider changes
      input.addEventListener('input', () => {
        valueSpan.textContent = input.value;
      });
      
      paramGroup.appendChild(valueSpan);
    } else if (param.type === 'select') {
      input = document.createElement('select');
      input.className = 'hyperparam-input';
      input.dataset.param = param.name;
      
      param.options.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt;
        option.textContent = opt;
        option.selected = opt === param.default;
        input.appendChild(option);
      });
    }
    
    paramGroup.appendChild(label);
    paramGroup.appendChild(input);
    hyperparamsContainer.appendChild(paramGroup);
  });
}

/**
 * Update playground visualization
 */
function updatePlaygroundVisualization() {
  const visualizationContainer = DOM.playgroundForm.querySelector('.playground-visualization');
  if (!visualizationContainer) return;
  
  visualizationContainer.innerHTML = `
    <div class="visualization-loading">
      <i class="fas fa-spinner fa-spin"></i>
      <p>Preparing ${STATE.playground.activeAlgorithm} visualization with ${STATE.playground.activeDataset} dataset...</p>
    </div>
  `;
  
  // Simulate loading
  setTimeout(() => {
    renderPlaygroundVisualization();
  }, 1000);
}

/**
 * Render playground visualization
 */
function renderPlaygroundVisualization() {
  const visualizationContainer = DOM.playgroundForm.querySelector('.playground-visualization');
  const algorithm = ALGORITHMS.find(a => a.id === STATE.playground.activeAlgorithm);
  
  if (!algorithm || !visualizationContainer) return;
  
  visualizationContainer.innerHTML = `
    <div class="playground-canvas" id="playground-canvas">
      <!-- Visualization would be rendered here by visualization library -->
      <div class="placeholder-visualization">
        <h4>${algorithm.title} on ${STATE.playground.activeDataset} Data</h4>
        <p>Visualization would show here with current parameters:</p>
        <ul>
          ${Object.entries(STATE.playground.hyperparameters).map(([key, value]) => `
          <li><strong>${key}</strong>: ${value}</li>
          `).join('')}
        </ul>
      </div>
    </div>
  `;
}

/**
 * Run playground training
 */
function runPlaygroundTraining() {
  if (STATE.playground.isTraining) return;
  
  STATE.playground.isTraining = true;
  STATE.playground.trainingProgress = 0;
  
  const trainBtn = DOM.playgroundForm.querySelector('.train-btn');
  trainBtn.disabled = true;
  trainBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Training...';
  
  const progressBar = DOM.playgroundForm.querySelector('.training-progress');
  progressBar.style.width = '0%';
  
  // Simulate training with progress updates
  const trainingInterval = setInterval(() => {
    STATE.playground.trainingProgress += Math.random() * 10;
    if (STATE.playground.trainingProgress >= 100) {
      STATE.playground.trainingProgress = 100;
      clearInterval(trainingInterval);
      trainingComplete();
    }
    
    progressBar.style.width = `${STATE.playground.trainingProgress}%`;
    progressBar.textContent = `${Math.floor(STATE.playground.trainingProgress)}%`;
  }, 300);
}

/**
 * Handle training completion
 */
function trainingComplete() {
  STATE.playground.isTraining = false;
  
  const trainBtn = DOM.playgroundForm.querySelector('.train-btn');
  trainBtn.disabled = false;
  trainBtn.innerHTML = '<i class="fas fa-play"></i> Train';
  
  // Show results
  STATE.playground.results = {
    accuracy: (Math.random() * 0.5 + 0.5).toFixed(2),
    loss: (Math.random() * 0.3).toFixed(4),
    time: (Math.random() * 3 + 1).toFixed(2)
  };
  
  showPlaygroundResults();
}

/**
 * Show playground training results
 */
function showPlaygroundResults() {
  const resultsContainer = DOM.playgroundForm.querySelector('.playground-results');
  if (!resultsContainer || !STATE.playground.results) return;
  
  resultsContainer.innerHTML = `
    <div class="results-header">
      <h4>Training Results</h4>
    </div>
    <div class="results-metrics">
      <div class="metric">
        <div class="metric-value">${STATE.playground.results.accuracy}</div>
        <div class="metric-label">Accuracy</div>
      </div>
      <div class="metric">
        <div class="metric-value">${STATE.playground.results.loss}</div>
        <div class="metric-label">Loss</div>
      </div>
      <div class="metric">
        <div class="metric-value">${STATE.playground.results.time}s</div>
        <div class="metric-label">Time</div>
      </div>
    </div>
    <div class="results-actions">
      <button class="btn-secondary">
        <i class="fas fa-chart-line"></i> View Metrics
      </button>
      <button class="btn-primary">
        <i class="fas fa-download"></i> Export Model
      </button>
    </div>
  `;
}

// =============================================
// SECTION: Projects Functionality
// =============================================

/**
 * Initialize projects section
 */
function initProjects() {
  if (!DOM.projectsSection) return;
  
  // Load saved projects
  const savedProjects = localStorage.getItem(CONFIG.storageKeys.userProjects);
  if (savedProjects) {
    try {
      STATE.userProjects = JSON.parse(savedProjects);
    } catch (e) {
      console.error('Failed to parse projects data', e);
    }
  }
  
  // Generate projects grid
  updateProjectsGrid();
}

/**
 * Update projects grid display
 */
function updateProjectsGrid() {
  const projectsGrid = DOM.projectsSection.querySelector('.projects-grid');
  if (!projectsGrid) return;
  
  projectsGrid.innerHTML = '';
  
  // Add algorithm projects
  ALGORITHMS.forEach(algorithm => {
    algorithm.projects.forEach((project, index) => {
      const projectId = `${algorithm.id}-project-${index}`;
      const isCompleted = STATE.completedProjects.includes(projectId);
      
      const projectCard = document.createElement('div');
      projectCard.className = `project-card ${isCompleted ? 'completed' : ''}`;
      projectCard.dataset.projectId = projectId;
      
      projectCard.innerHTML = `
        <div class="project-badge ${project.difficulty}">
          ${CONFIG.difficulties[project.difficulty].name}
        </div>
        <div class="project-content">
          <h5>${project.title}</h5>
          <p>${algorithm.title} Project</p>
          <div class="project-status">
            ${isCompleted ? `
            <span class="completed-badge">
              <i class="fas fa-check-circle"></i> Completed
            </span>
            ` : `
            <span class="incomplete-badge">
              <i class="fas fa-circle"></i> Incomplete
            </span>
            `}
          </div>
        </div>
      `;
      
      projectsGrid.appendChild(projectCard);
    });
  });
  
  // Add user projects
  STATE.userProjects.forEach(project => {
    const projectCard = document.createElement('div');
    projectCard.className = `project-card ${project.completed ? 'completed' : ''}`;
    projectCard.dataset.projectId = project.id;
    
    projectCard.innerHTML = `
      <div class="project-badge ${project.difficulty || 'intermediate'}">
        ${project.difficulty ? CONFIG.difficulties[project.difficulty].name : 'Custom'}
      </div>
      <div class="project-content">
        <h5>${project.title}</h5>
        <p>Custom Project</p>
        <div class="project-status">
          ${project.completed ? `
          <span class="completed-badge">
            <i class="fas fa-check-circle"></i> Completed
          </span>
          ` : `
          <span class="incomplete-badge">
            <i class="fas fa-circle"></i> Incomplete
          </span>
          `}
        </div>
      </div>
    `;
    
    projectsGrid.appendChild(projectCard);
  });
}

/**
 * Show project detail view
 * @param {string} projectId - ID of project to show
 */
function showProjectDetail(projectId) {
  // Check if it's a predefined algorithm project
  const algorithmProjectMatch = projectId.match(/^(.+)-project-(\d+)$/);
  
  if (algorithmProjectMatch) {
    const algorithmId = algorithmProjectMatch[1];
    const projectIndex = parseInt(algorithmProjectMatch[2]);
    const algorithm = ALGORITHMS.find(a => a.id === algorithmId);
    
    if (algorithm && algorithm.projects[projectIndex]) {
      showAlgorithmProjectDetail(algorithm, algorithm.projects[projectIndex], projectId);
      return;
    }
  }
  
  // Otherwise it's a user project
  const project = STATE.userProjects.find(p => p.id === projectId);
  if (project) {
    showUserProjectDetail(project);
  }
}

/**
 * Show algorithm project detail
 * @param {Object} algorithm - Algorithm data
 * @param {Object} project - Project data
 * @param {string} projectId - Full project ID
 */
function showAlgorithmProjectDetail(algorithm, project, projectId) {
  const isCompleted = STATE.completedProjects.includes(projectId);
  
  const modal = document.createElement('div');
  modal.className = 'project-modal';
  modal.innerHTML = `
    <div class="modal-content">
      <div class="modal-header">
        <h3>${project.title}</h3>
        <button class="close-modal">
          <i class="fas fa-times"></i>
        </button>
      </div>
      <div class="modal-body">
        <div class="project-meta">
          <span class="badge ${project.difficulty}">
            ${CONFIG.difficulties[project.difficulty].name}
          </span>
          <span class="algorithm-badge">
            <i class="fas fa-project-diagram"></i> ${algorithm.title}
          </span>
          <span class="xp-badge">
            <i class="fas fa-star"></i> ${project.xp} XP
          </span>
        </div>
        
        <div class="project-description">
          <h4>Description</h4>
          <p>${project.description}</p>
        </div>
        
        <div class="project-steps">
          <h4>Steps</h4>
          <ol>
            ${project.steps.map(step => `
            <li>${step}</li>
            `).join('')}
          </ol>
        </div>
        
        <div class="project-resources">
          <h4>Resources</h4>
          <ul>
            <li>
              <a href="#">
                <i class="fas fa-file-code"></i> Starter Notebook
              </a>
            </li>
            <li>
              <a href="#">
                <i class="fas fa-database"></i> Dataset Download
              </a>
            </li>
            <li>
              <a href="#">
                <i class="fas fa-book"></i> Documentation
              </a>
            </li>
          </ul>
        </div>
      </div>
      <div class="modal-footer">
        ${isCompleted ? `
        <button class="btn-secondary">
          <i class="fas fa-redo"></i> Redo Project
        </button>
        ` : `
        <button class="btn-primary complete-project" data-project-id="${projectId}">
          <i class="fas fa-check"></i> Mark as Complete
        </button>
        `}
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  
  // Add event listeners
  modal.querySelector('.close-modal').addEventListener('click', () => {
    modal.remove();
  });
  
  if (!isCompleted) {
    modal.querySelector('.complete-project').addEventListener('click', () => {
      completeProject(projectId);
      modal.remove();
    });
  }
}

/**
 * Show user project detail
 * @param {Object} project - Project data
 */
function showUserProjectDetail(project) {
  const modal = document.createElement('div');
  modal.className = 'project-modal';
  modal.innerHTML = `
    <div class="modal-content">
      <div class="modal-header">
        <h3>${project.title}</h3>
        <button class="close-modal">
          <i class="fas fa-times"></i>
        </button>
      </div>
      <div class="modal-body">
        <div class="project-meta">
          <span class="badge ${project.difficulty || 'intermediate'}">
            ${project.difficulty ? CONFIG.difficulties[project.difficulty].name : 'Custom'}
          </span>
          <span class="xp-badge">
            <i class="fas fa-star"></i> ${project.xp || 200} XP
          </span>
        </div>
        
        <div class="project-description">
          <h4>Description</h4>
          <p>${project.description || 'No description provided.'}</p>
        </div>
        
        <div class="project-notes">
          <h4>Your Notes</h4>
          <textarea class="project-notes-input" placeholder="Add your project notes here...">${project.notes || ''}</textarea>
        </div>
      </div>
      <div class="modal-footer">
        ${project.completed ? `
        <button class="btn-secondary">
          <i class="fas fa-redo"></i> Redo Project
        </button>
        ` : `
        <button class="btn-primary complete-project" data-project-id="${project.id}">
          <i class="fas fa-check"></i> Mark as Complete
        </button>
        `}
        <button class="btn-danger delete-project" data-project-id="${project.id}">
          <i class="fas fa-trash"></i> Delete
        </button>
      </div>
    </div>
  `;
  
  document.body.appendChild(modal);
  
  // Add event listeners
  modal.querySelector('.close-modal').addEventListener('click', () => {
    saveProjectNotes(project.id, modal.querySelector('.project-notes-input').value);
    modal.remove();
  });
  
  if (!project.completed) {
    modal.querySelector('.complete-project').addEventListener('click', () => {
      completeProject(project.id);
      modal.remove();
    });
  }
  
  modal.querySelector('.delete-project').addEventListener('click', () => {
    if (confirm('Are you sure you want to delete this project?')) {
      deleteProject(project.id);
      modal.remove();
    }
  });
}

/**
 * Handle project form submission
 * @param {Event} e - Form submit event
 */
function handleProjectSubmit(e) {
  e.preventDefault();
  
  const form = e.target;
  const title = form.querySelector('#projectTitle').value;
  const description = form.querySelector('#projectDescription').value;
  const difficulty = form.querySelector('#projectDifficulty').value;
  
  const newProject = {
    id: `project-${Date.now()}`,
    title,
    description,
    difficulty,
    xp: difficulty === 'beginner' ? 200 : difficulty === 'intermediate' ? 300 : 400,
    completed: false,
    createdAt: new Date().toISOString()
  };
  
  STATE.userProjects.push(newProject);
  saveUserProjects();
  updateProjectsGrid();
  
  // Reset form
  form.reset();
  
  // Show success message
  showNotification('Project created successfully!', 'success');
}

/**
 * Save project notes
 * @param {string} projectId - Project ID
 * @param {string} notes - Project notes
 */
function saveProjectNotes(projectId, notes) {
  const project = STATE.userProjects.find(p => p.id === projectId);
  if (project) {
    project.notes = notes;
    saveUserProjects();
  }
}

/**
 * Complete a project
 * @param {string} projectId - Project ID to complete
 */
function completeProject(projectId) {
  // Check if it's a predefined project
  if (projectId.includes('-project-')) {
    if (!STATE.completedProjects.includes(projectId)) {
      STATE.completedProjects.push(projectId);
      
      // Find the project to get XP value
      const algorithmProjectMatch = projectId.match(/^(.+)-project-(\d+)$/);
      if (algorithmProjectMatch) {
        const algorithmId = algorithmProjectMatch[1];
        const projectIndex = parseInt(algorithmProjectMatch[2]);
        const algorithm = ALGORITHMS.find(a => a.id === algorithmId);
        
        if (algorithm && algorithm.projects[projectIndex]) {
          updateXp(algorithm.projects[projectIndex].xp);
        }
      }
      
      saveUserProgress();
      updateProjectsGrid();
      showNotification('Project completed! XP earned.', 'success');
    }
  } else {
    // It's a user project
    const project = STATE.userProjects.find(p => p.id === projectId);
    if (project && !project.completed) {
      project.completed = true;
      updateXp(project.xp || 200);
      saveUserProjects();
      updateProjectsGrid();
      showNotification('Project completed! XP earned.', 'success');
    }
  }
}

/**
 * Delete a project
 * @param {string} projectId - Project ID to delete
 */
function deleteProject(projectId) {
  STATE.userProjects = STATE.userProjects.filter(p => p.id !== projectId);
  saveUserProjects();
  updateProjectsGrid();
}

/**
 * Save user projects to local storage
 */
function saveUserProjects() {
  localStorage.setItem(CONFIG.storageKeys.userProjects, JSON.stringify(STATE.userProjects));
}

// =============================================
// SECTION: Quiz & Progress Tracking
// =============================================

/**
 * Handle quiz submission
 * @param {Event} e - Click event
 */
function handleQuizSubmit(e) {
  const quizContainer = e.target.closest('.quiz-container');
  const questions = quizContainer.querySelectorAll('.question-card');
  let correctAnswers = 0;
  let totalQuestions = questions.length;
  
  // Validate each question
  questions.forEach((question, index) => {
    const selectedOption = question.querySelector('input[type="radio"]:checked');
    const questionNumber = index + 1;
    
    if (selectedOption) {
      const isCorrect = selectedOption.getAttribute('data-correct') === "true";
      
      if (isCorrect) {
        correctAnswers++;
        selectedOption.parentElement.classList.add('correct');
      } else {
        selectedOption.parentElement.classList.add('incorrect');
        // Highlight correct answer
        question.querySelector('input[type="radio"][data-correct="true"]')
          .parentElement.classList.add('correct');
      }
    } else {
      // No answer selected
      question.classList.add('unanswered');
    }
  });
  
  // Calculate score
  const score = Math.round((correctAnswers / totalQuestions) * 100);
  const algorithm = STATE.currentAlgorithm;
  
  // Update progress
  updateQuizProgress(algorithm, score);
  
  // Show results
  showQuizResults(quizContainer, score, correctAnswers, totalQuestions);
  
  // Disable further answers
  quizContainer.querySelectorAll('input[type="radio"]').forEach((input) => {
    input.disabled = true;
  });
  
  // Update XP and level
  updateXp(score * 2); // More points for better scores
}

/**
 * Show quiz results
 * @param {HTMLElement} container - Quiz container
 * @param {number} score - Quiz score percentage
 * @param {number} correct - Number of correct answers
 * @param {number} total - Total number of questions
 */
function showQuizResults(container, score, correct, total) {
  // Remove existing results if present
  const existingResults = container.querySelector('.quiz-results');
  if (existingResults) existingResults.remove();
  
  const resultsElement = document.createElement('div');
  resultsElement.className = 'quiz-results';
  
  // Determine result message
  let resultMessage = '';
  if (score >= 90) {
    resultMessage = 'Excellent! You\'ve mastered this concept.';
  } else if (score >= 70) {
    resultMessage = 'Good job! You understand the main concepts.';
  } else if (score >= 50) {
    resultMessage = 'Not bad! Review the material and try again.';
  } else {
    resultMessage = 'Keep practicing! Review the algorithm details.';
  }
  
  resultsElement.innerHTML = `
    <h4>Quiz Results: ${score}%</h4>
    <p>You answered ${correct} out of ${total} questions correctly</p>
    <div class="quiz-result-message ${score >= 70 ? 'success' : 'info'}">
      ${resultMessage}
    </div>
    ${score >= 70 ? '<div class="quiz-badge"><i class="fas fa-trophy"></i> Algorithm Mastered</div>' : ''}
  `;
  
  container.appendChild(resultsElement);
  
  // Scroll to results
  setTimeout(() => {
    resultsElement.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }, 100);
}

/**
 * Reset quiz to initial state
 * @param {Event} e - Click event
 */
function resetQuiz(e) {
  const quizContainer = e.target.closest('.quiz-container');
  
  // Remove results if present
  const results = quizContainer.querySelector('.quiz-results');
  if (results) results.remove();
  
  // Reset all questions
  quizContainer.querySelectorAll('.question-card').forEach((question) => {
    question.classList.remove('unanswered');
    question.querySelectorAll('.option').forEach((option) => {
      option.classList.remove('correct', 'incorrect');
    });
  });
  
  // Clear selections and re-enable
  quizContainer.querySelectorAll('input[type="radio"]').forEach((input) => {
    input.checked = false;
    input.disabled = false;
  });
}

/**
 * Update quiz progress for an algorithm
 * @param {string} algorithm - Algorithm ID
 * @param {number} score - Quiz score
 */
function updateQuizProgress(algorithm, score) {
  if (!STATE.quizProgress[algorithm] || score > STATE.quizProgress[algorithm].score) {
    STATE.quizProgress[algorithm] = {
      score: score,
      timestamp: new Date().toISOString()
    };
    
    // Update streak
    const today = new Date().toDateString();
    if (score >= 70) {
      if (STATE.streakDate !== today) {
        STATE.streak++;
        STATE.streakDate = today;
        checkStreakAchievement();
      }
    } else {
      STATE.streak = 0;
      STATE.streakDate = null;
    }
    
    // Save progress
    saveUserProgress();
    updateRoadmapProgress();
  }
}

/**
 * Update user XP
 * @param {number} points - XP points to add
 */
function updateXp(points) {
  STATE.xp += points;
  
  // Check for level up
  const xpNeeded = STATE.level * CONFIG.xpValues.levelMultiplier;
  if (STATE.xp >= xpNeeded) {
    STATE.level++;
    STATE.xp = STATE.xp - xpNeeded;
    showLevelUpNotification();
  }
  
  // Update UI
  updateXpBar();
  saveUserProgress();
}

/**
 * Show level up notification
 */
function showLevelUpNotification() {
  const notification = document.createElement('div');
  notification.className = 'level-up-notification';
  notification.innerHTML = `
    <div class="notification-content">
      <div class="level-up-icon">
        <i class="fas fa-trophy"></i>
      </div>
      <h3>Level Up!</h3>
      <p>You've reached level ${STATE.level}</p>
      <div class="xp-progress">
        <div class="progress-bar">
          <div class="progress-fill" style="width: 0%"></div>
        </div>
        <span>0/${STATE.level * CONFIG.xpValues.levelMultiplier} XP</span>
      </div>
      <button class="btn-primary close-notification">Continue Learning</button>
    </div>
  `;
  
  document.body.appendChild(notification);
  
  // Animate progress bar
  setTimeout(() => {
    const progressFill = notification.querySelector('.progress-fill');
    const percentage = (STATE.xp / (STATE.level * CONFIG.xpValues.levelMultiplier)) * 100;
    progressFill.style.width = `${percentage}%`;
  }, 100);
  
  // Close button
  notification.querySelector('.close-notification').addEventListener('click', () => {
    notification.classList.add('fade-out');
    setTimeout(() => notification.remove(), 300);
  });
  
  // Auto-close after 5 seconds
  setTimeout(() => {
    if (notification.parentNode) {
      notification.classList.add('fade-out');
      setTimeout(() => notification.remove(), 300);
    }
  }, 5000);
}

// =============================================
// SECTION: User Data & Progress
// =============================================

/**
 * Load user data from local storage
 */
function loadUserData() {
  // Load progress
  const savedProgress = localStorage.getItem(CONFIG.storageKeys.progress);
  if (savedProgress) {
    try {
      const progress = JSON.parse(savedProgress);
      STATE.quizProgress = progress.quizProgress || {};
      STATE.xp = progress.xp || 0;
      STATE.level = progress.level || 1;
      STATE.streak = progress.streak || 0;
      STATE.streakDate = progress.streakDate || null;
      STATE.completedAlgorithms = progress.completedAlgorithms || [];
      STATE.completedProjects = progress.completedProjects || [];
    } catch (e) {
      console.error('Failed to parse progress data', e);
    }
  }
  
  // Load achievements
  const savedAchievements = localStorage.getItem(CONFIG.storageKeys.achievements);
  if (savedAchievements) {
    try {
      STATE.userAchievements = JSON.parse(savedAchievements);
    } catch (e) {
      console.error('Failed to parse achievements data', e);
    }
  }
  
  // Load theme
  const savedTheme = localStorage.getItem(CONFIG.storageKeys.theme);
  if (savedTheme) {
    STATE.currentTheme = savedTheme;
    applyTheme(savedTheme);
  }
  
  // Load dark mode preference
  const savedDarkMode = localStorage.getItem('darkMode');
  if (savedDarkMode) {
    STATE.isDarkMode = savedDarkMode === 'true';
    toggleDarkMode(STATE.isDarkMode);
  }
  
  // Update UI
  updateXpBar();
  updateRoadmapProgress();
}

/**
 * Save user progress to local storage
 */
function saveUserProgress() {
  const progressData = {
    quizProgress: STATE.quizProgress,
    xp: STATE.xp,
    level: STATE.level,
    streak: STATE.streak,
    streakDate: STATE.streakDate,
    completedAlgorithms: STATE.completedAlgorithms,
    completedProjects: STATE.completedProjects,
    timestamp: new Date().toISOString()
  };
  
  localStorage.setItem(CONFIG.storageKeys.progress, JSON.stringify(progressData));
  localStorage.setItem(CONFIG.storageKeys.achievements, JSON.stringify(STATE.userAchievements));
  localStorage.setItem(CONFIG.storageKeys.theme, STATE.currentTheme);
  localStorage.setItem('darkMode', STATE.isDarkMode);
}

/**
 * Update XP bar in UI
 */
function updateXpBar() {
  const xpBar = document.querySelector('.xp-bar');
  if (!xpBar) return;
  
  const xpNeeded = STATE.level * CONFIG.xpValues.levelMultiplier;
  const percentage = Math.min(100, (STATE.xp / xpNeeded) * 100);
  
  xpBar.querySelector('.xp-fill').style.width = `${percentage}%`;
  xpBar.querySelector('.xp-text').textContent = `Level ${STATE.level} | ${STATE.xp}/${xpNeeded} XP`;
  
  // Update level indicator
  const levelIndicator = xpBar.querySelector('.level-indicator');
  if (levelIndicator) {
    levelIndicator.textContent = STATE.level;
  }
}

/**
 * Update roadmap progress indicators
 */
function updateRoadmapProgress() {
  DOM.timelineItems.forEach((item) => {
    const algorithmItems = item.querySelectorAll('.item-algorithms li[data-algorithm]');
    let completed = 0;
    
    algorithmItems.forEach((li) => {
      const algorithm = li.dataset.algorithm;
      if (STATE.quizProgress[algorithm] && STATE.quizProgress[algorithm].score >= 70) {
        completed++;
        li.querySelector('i').className = 'fas fa-check-circle';
        li.classList.add('completed');
      } else {
        li.querySelector('i').className = 'far fa-circle';
        li.classList.remove('completed');
      }
    });
    
    const total = algorithmItems.length;
    const percentage = total > 0 ? Math.round((completed / total) * 100) : 0;
    
    const progressText = item.querySelector('.progress-text');
    const progressBar = item.querySelector('.progress-fill');
    
    if (progressText && progressBar) {
      progressText.textContent = `${completed} of ${total} completed`;
      progressBar.style.width = `${percentage}%`;
    }
  });
}

/**
 * Track algorithm view for progress
 * @param {string} algorithmId - Algorithm ID
 */
function trackAlgorithmView(algorithmId) {
  if (!STATE.completedAlgorithms.includes(algorithmId)) {
    STATE.completedAlgorithms.push(algorithmId);
    updateXp(CONFIG.xpValues.algorithmView);
    saveUserProgress();
  }
}


/**
 * Check for streak achievements
 */
function checkStreakAchievement() {
  const streakMilestones = [3, 5, 7, 10, 14, 21, 30];
  
  streakMilestones.forEach((milestone) => {
    if (STATE.streak === milestone && !STATE.userAchievements.includes(`Streak ${milestone}`)) {
      STATE.userAchievements.push(`Streak ${milestone}`);
      showAchievementNotification(`Streak ${milestone}`, `Completed ${milestone} quizzes in a row!`);
      saveUserProgress();
    }
  });
}

/**
 * Show achievement notification
 * @param {string} title - Achievement title
 * @param {string} description - Achievement description
 */
function showAchievementNotification(title, description) {
  const notification = document.createElement('div');
  notification.className = 'achievement-notification';
  notification.innerHTML = `
    <div class="notification-content">
      <div class="achievement-icon">
        <i class="fas fa-trophy"></i>
      </div>
      <h3>Achievement Unlocked!</h3>
      <h4>${title}</h4>
      <p>${description}</p>
      <button class="btn-primary close-notification">Awesome!</button>
    </div>
  `;
  
  document.body.appendChild(notification);
  
  notification.querySelector('.close-notification').addEventListener('click', () => {
    notification.remove();
  });
  
  // Auto-close after 5 seconds
  setTimeout(() => {
    if (notification.parentNode) {
      notification.remove();
    }
  }, 5000);
}

/**
 * Show general notification
 * @param {string} message - Notification message
 * @param {string} type - Notification type (success, error, info)
 */
function showNotification(message, type = 'info') {
  const notification = document.createElement('div');
  notification.className = `notification ${type}`;
  notification.innerHTML = `
    <div class="notification-content">
      <p>${message}</p>
    </div>
  `;
  
  document.body.appendChild(notification);
  
  // Auto-remove after 3 seconds
  setTimeout(() => {
    notification.classList.add('fade-out');
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// =============================================
// SECTION: UI & Theme Management
// =============================================

/**
 * Initialize theme and UI settings
 */
function initUI() {
  // Check for saved theme preference
  const savedTheme = localStorage.getItem(CONFIG.storageKeys.theme);
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  
  if (savedTheme) {
    applyTheme(savedTheme);
  } else if (prefersDark) {
    toggleDarkMode(true);
  }
  
  // Initialize animations
  initAnimations();
}

/**
 * Apply selected theme
 * @param {string} themeName - Theme name to apply
 */
function applyTheme(themeName) {
  if (!CONFIG.colorThemes[themeName]) {
    console.warn(`Theme ${themeName} not found, using default`);
    themeName = 'default';
  }
  
  STATE.currentTheme = themeName;
  localStorage.setItem(CONFIG.storageKeys.theme, themeName);
  
  const theme = CONFIG.colorThemes[themeName];
  
  // Apply CSS variables
  document.documentElement.style.setProperty('--primary', theme.primary);
  document.documentElement.style.setProperty('--secondary', theme.secondary);
  document.documentElement.style.setProperty('--accent', theme.accent);
  document.documentElement.style.setProperty('--text', theme.text);
  document.documentElement.style.setProperty('--background', theme.background);
  
  // Update UI elements
  document.querySelectorAll('.neon-logo, .logo-svg').forEach((el) => {
    el.style.textShadow = `0 0 10px ${theme.accent}, 0 0 20px ${theme.accent}, 0 0 30px ${theme.accent}`;
  });
  
  // Special handling for terminal theme
  if (themeName === 'terminal') {
    document.body.classList.add('terminal-mode');
  } else {
    document.body.classList.remove('terminal-mode');
  }
}

/**
 * Cycle through available themes
 */
function cycleTheme() {
  const themeNames = Object.keys(CONFIG.colorThemes);
  const currentIndex = themeNames.indexOf(STATE.currentTheme);
  const nextIndex = (currentIndex + 1) % themeNames.length;
  
  applyTheme(themeNames[nextIndex]);
}

/**
 * Toggle dark mode
 * @param {boolean} [forceState] - Optional forced state
 */
function toggleDarkMode(forceState) {
  STATE.isDarkMode = forceState !== undefined ? forceState : !STATE.isDarkMode;
  
  if (STATE.isDarkMode) {
    document.body.classList.add('dark-mode');
  } else {
    document.body.classList.remove('dark-mode');
  }
  
  localStorage.setItem('darkMode', STATE.isDarkMode);
}

// =============================================
// SECTION: Animations & Effects
// =============================================

/**
 * Initialize animations
 */
function initAnimations() {
  // Hero section animation
  animateHeroSection();
  
  // Scroll-triggered animations
  initScrollAnimations();
  
  // Neural network animation
  animateNeuralNetwork();
}

/**
 * Animate hero section elements
 */
function animateHeroSection() {
  const titleLines = document.querySelectorAll('.hero-title .title-line');
  const subtitle = document.querySelector('.hero-subtitle');
  const neuralNetwork = document.querySelector('.neural-network-animation');
  
  // Reset initial state
  gsap.set([titleLines, subtitle, neuralNetwork], { opacity: 0, y: 20 });
  
  // Animate title lines sequentially
  gsap.to(titleLines[0], {
    opacity: 1,
    y: 0,
    duration: 0.8,
    ease: 'power2.out'
  });
  
  gsap.to(titleLines[1], {
    opacity: 1,
    y: 0,
    duration: 0.8,
    delay: 0.3,
    ease: 'power2.out',
    onComplete: animateNeuralNetwork
  });
  
  // Animate subtitle
  gsap.to(subtitle, {
    opacity: 1,
    y: 0,
    duration: 0.6,
    delay: 0.8,
    ease: 'power2.out'
  });
  
  // Animate buttons
  gsap.to('.hero-actions a', {
    opacity: 1,
    y: 0,
    duration: 0.5,
    delay: 1,
    stagger: 0.1,
    ease: 'back.out'
  });
}

/**
 * Animate neural network visualization
 */
function animateNeuralNetwork() {
  const neurons = document.querySelectorAll('.neuron');
  const connections = document.querySelectorAll('.connection');
  
  // Animate neurons
  gsap.from(neurons, {
    scale: 0,
    opacity: 0,
    duration: 0.6,
    stagger: 0.05,
    ease: 'back.out'
  });
  
  // Animate connections with delay
  gsap.from(connections, {
    scaleX: 0,
    opacity: 0,
    duration: 0.8,
    delay: 0.5,
    stagger: 0.03,
    ease: 'power2.out',
    transformOrigin: 'left center'
  });
  
  // Continuous pulse animation for neurons
  neurons.forEach((neuron) => {
    gsap.to(neuron, {
      scale: 1.1,
      duration: 1.5,
      repeat: -1,
      yoyo: true,
      ease: 'sine.inOut'
    });
  });
  
  // Continuous flow animation for connections
  connections.forEach((connection) => {
    gsap.to(connection, {
      backgroundPositionX: '100%',
      duration: 3,
      repeat: -1,
      ease: 'none'
    });
  });
}

/**
 * Initialize scroll-triggered animations
 */
function initScrollAnimations() {
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('in-view');
      }
    });
  }, { threshold: CONFIG.scrollThreshold });
  
  // Observe elements that should animate on scroll
  document.querySelectorAll('.algorithm-card, .section-header, .timeline-item, .resource-card').forEach((el) => {
    observer.observe(el);
  });
}

// =============================================
// SECTION: Utility Functions
// =============================================

/**
 * Debounce function to limit how often a function is called
 * @param {Function} func - Function to debounce
 * @param {number} wait - Wait time in milliseconds
 * @return {Function} - Debounced function
 */
function debounce(func, wait) {
  let timeout;
  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

/**
 * Handle scroll events
 */
function handleScroll() {
  STATE.scrollPosition = window.scrollY;
  STATE.scrollDirection = STATE.scrollPosition > STATE.lastScrollPosition ? 'down' : 'up';
  STATE.lastScrollPosition = STATE.scrollPosition;
  
  updateNavbarState();
  updateSectionVisibility();
  updateScrollProgress();
  updateTimelineProgress();
}

// Toggle mobile menu
const menuToggle = document.querySelector('.menu-toggle');
const navLinks = document.querySelector('.nav-links ul');

menuToggle.addEventListener('click', () => {
  navLinks.classList.toggle('show');
});

const canvas = document.getElementById("neural-canvas");
const ctx = canvas.getContext("2d");
let width = canvas.width = window.innerWidth;
let height = canvas.height = 80;

const nodes = [];
const nodeCount = 30;
const maxDist = 120;

// Create nodes
for(let i=0;i<nodeCount;i++){
    nodes.push({
        x: Math.random() * width,
        y: Math.random() * height,
        vx: (Math.random()-0.5)*0.5,
        vy: (Math.random()-0.5)*0.5,
        radius: 2 + Math.random()*3
    });
}

// Draw nodes & lines
function animate(){
    ctx.clearRect(0,0,width,height);

    // Update nodes
    nodes.forEach(node=>{
        node.x += node.vx;
        node.y += node.vy;

        if(node.x<0 || node.x>width) node.vx *= -1;
        if(node.y<0 || node.y>height) node.vy *= -1;

        // Draw node
        ctx.beginPath();
        ctx.arc(node.x,node.y,node.radius,0,Math.PI*2);
        ctx.fillStyle = "rgba(123,44,191,0.9)";
        ctx.fill();
        ctx.closePath();
    });

    // Connect nodes if close
    for(let i=0;i<nodeCount;i++){
        for(let j=i+1;j<nodeCount;j++){
            let dx = nodes[i].x - nodes[j].x;
            let dy = nodes[i].y - nodes[j].y;
            let dist = Math.sqrt(dx*dx + dy*dy);
            if(dist<maxDist){
                ctx.beginPath();
                ctx.moveTo(nodes[i].x, nodes[i].y);
                ctx.lineTo(nodes[j].x, nodes[j].y);
                ctx.strokeStyle = `rgba(157,78,221, ${1-dist/maxDist})`;
                ctx.lineWidth = 1;
                ctx.stroke();
                ctx.closePath();
            }
        }
    }

    requestAnimationFrame(animate);
}

animate();

// Handle resize
window.addEventListener("resize",()=>{
    width = canvas.width = window.innerWidth;
    height = canvas.height = 80;
});


// Scroll cinematic effect
const navbar = document.querySelector('.cinematic-navbar');
window.addEventListener('scroll', () => {
  if (window.scrollY > 50) {
    navbar.classList.add('scrolled');
  } else {
    navbar.classList.remove('scrolled');
  }
});


/**
 * Update navbar state based on scroll
 */
function updateNavbarState() {
  if (STATE.scrollPosition > 100) {
    DOM.navbar.classList.add('scrolled');
  } else {
    DOM.navbar.classList.remove('scrolled');
  }
  
  // Update active nav link based on current section
  const sections = document.querySelectorAll('section');
  let newActiveSection = null;
  
  sections.forEach((section) => {
    const rect = section.getBoundingClientRect();
    const sectionTop = rect.top + window.scrollY;
    const sectionHeight = rect.height;
    
    if (window.scrollY >= sectionTop - 200 && 
        window.scrollY < sectionTop + sectionHeight - 200) {
      newActiveSection = section.getAttribute('id');
    }
  });
  
  // Only update if section changed
  if (newActiveSection && newActiveSection !== STATE.activeSection) {
    STATE.activeSection = newActiveSection;
    
    document.querySelectorAll('.nav-link').forEach((link) => {
      link.classList.remove('active');
      if (link.getAttribute('href') === `#${newActiveSection}`) {
        link.classList.add('active');
      }
    });
  }
}

/**
 * Update section visibility state
 */
function updateSectionVisibility() {
  const sections = document.querySelectorAll('section');
  sections.forEach((section) => {
    const rect = section.getBoundingClientRect();
    const isVisible = rect.top < window.innerHeight * 0.7 && rect.bottom > 0;
    
    if (isVisible) {
      section.classList.add('active');
    } else {
      section.classList.remove('active');
    }
  });
}

/**
 * Update scroll progress indicator
 */
function updateScrollProgress() {
  const scrollProgress = document.querySelector('.scroll-progress');
  if (!scrollProgress) return;
  
  const scrollHeight = document.documentElement.scrollHeight - window.innerHeight;
  const scrollPercentage = (window.scrollY / scrollHeight) * 100;
  
  scrollProgress.style.width = `${scrollPercentage}%`;
}

/**
 * Update timeline progress indicators
 */
function updateTimelineProgress() {
  const timeline = document.querySelector('.roadmap-timeline');
  if (!timeline) return;
  
  const timelineItems = document.querySelectorAll('.timeline-item');
  timelineItems.forEach((item) => {
    const rect = item.getBoundingClientRect();
    const isVisible = rect.top < window.innerHeight * 0.8 && rect.bottom > 0;
    
    if (isVisible) {
      item.classList.add('active');
    }
  });
}

/**
 * Handle window resize events
 */
function handleResize() {
  updateSectionVisibility();
  updateTimelineProgress();
}

/**
 * Smooth scroll to section
 * @param {string} selector - Section selector
 */
function smoothScrollToSection(selector) {
  const target = document.querySelector(selector);
  if (target) {
    window.scrollTo({
      top: target.offsetTop - 80,
      behavior: 'smooth'
    });
  }
}

/**
 * Track Konami code input
 * @param {KeyboardEvent} e - Keydown event
 */
function trackKonamiCode(e) {
  if (STATE.konamiSequence[STATE.konamiCode.length] === e.key) {
    STATE.konamiCode.push(e.key);
    if (STATE.konamiCode.length === STATE.konamiSequence.length) {
      activateEasterEgg();
      STATE.konamiCode = [];
    }
  } else {
    STATE.konamiCode = [];
  }
}

/**
 * Activate Konami code easter egg
 */
function activateEasterEgg() {
  const easterEgg = document.createElement('div');
  easterEgg.className = 'easter-egg';
  easterEgg.innerHTML = `
    <div class="easter-egg-content">
      <h3>Secret Unlocked!</h3>
      <p>You found the Konami code easter egg!</p>
      <div class="egg-animation">
        <div class="egg"></div>
      </div>
      <button class="btn-primary close-egg">Close</button>
    </div>
  `;
  
  document.body.appendChild(easterEgg);
  
  // Animate egg
  gsap.to('.egg', {
    rotation: 360,
    duration: 2,
    repeat: -1,
    ease: 'none'
  });
  
  // Close button
  easterEgg.querySelector('.close-egg').addEventListener('click', () => {
    easterEgg.remove();
  });
}

/**
 * Handle navigation keyboard shortcuts
 * @param {KeyboardEvent} e - Keydown event
 */
function handleNavigationShortcuts(e) {
  switch (e.key) {
    case 'g': window.location.href = '#'; break;
    case 's': 
      if (DOM.searchInput) DOM.searchInput.focus(); 
      break;
    case 'h': window.location.href = '#dashboard'; break;
    case 'a': window.location.href = '#algorithms'; break;
    case 'r': window.location.href = '#roadmap'; break;
    case 'p': window.location.href = '#playground'; break;
    case 'j': window.location.href = '#projects'; break;
  }
}

/**
 * Check if user is first-time visitor
 */
function checkFirstTimeUser() {
  const firstVisit = !localStorage.getItem('visitedBefore');
  if (firstVisit) {
    showWelcomeTour();
    localStorage.setItem('visitedBefore', 'true');
  }
}

/**
 * Show welcome tour for first-time users
 */
function showWelcomeTour() {
  const tour = document.createElement('div');
  tour.className = 'welcome-tour';
  tour.innerHTML = `
    <div class="tour-content">
      <h3>Welcome to DataKarya!</h3>
      <p>Let's take a quick tour of the platform:</p>
      <div class="tour-steps">
        <div class="tour-step active">
          <i class="fas fa-search"></i>
          <p>Search for algorithms using the search bar</p>
        </div>
        <div class="tour-step">
          <i class="fas fa-project-diagram"></i>
          <p>Explore interactive visualizations</p>
        </div>
        <div class="tour-step">
          <i class="fas fa-code"></i>
          <p>View implementations in multiple languages</p>
        </div>
        <div class="tour-step">
          <i class="fas fa-road"></i>
          <p>Follow the learning path for structured progress</p>
        </div>
      </div>
      <div class="tour-actions">
        <button class="btn-secondary skip-tour">Skip Tour</button>
        <button class="btn-primary next-step">Next</button>
      </div>
    </div>
  `;
  
  document.body.appendChild(tour);
  
  let currentStep = 0;
  const steps = tour.querySelectorAll('.tour-step');
  
  // Next button
  tour.querySelector('.next-step').addEventListener('click', () => {
    steps[currentStep].classList.remove('active');
    currentStep = (currentStep + 1) % steps.length;
    
    if (currentStep === steps.length - 1) {
      tour.querySelector('.next-step').textContent = 'Finish';
    }
    
    if (currentStep === 0) {
      tour.remove();
    } else {
      steps[currentStep].classList.add('active');
    }
  });
  
  // Skip button
  tour.querySelector('.skip-tour').addEventListener('click', () => {
    tour.remove();
  });
}

/**
 * Set up service worker for PWA functionality
 */
function setupServiceWorker() {
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
      navigator.serviceWorker.register('/sw.js').then(registration => {
        console.log('ServiceWorker registration successful');
      }).catch(err => {
        console.log('ServiceWorker registration failed: ', err);
      });
    });
  }
}

// =============================================
// SECTION: Initialization
// =============================================

// Main initialization when DOM is loaded
document.addEventListener('DOMContentLoaded', initializeApp);

// Fallback initialization in case DOMContentLoaded fails
setTimeout(() => {
  if (!STATE.isInitialized) {
    console.warn('Fallback initialization triggered');
    initializeApp();
  }
}, CONFIG.maxPreloaderTimeout);

// Contact Form Functionality
document.addEventListener('DOMContentLoaded', function() {
  const contactForm = document.getElementById('contactForm');
  const formFields = contactForm ? contactForm.querySelectorAll('input, textarea, select') : [];
  const submitBtn = contactForm ? contactForm.querySelector('button[type="submit"]') : null;

  // Initialize form events if form exists
  if (contactForm) {
    // Add event listeners to all form fields
    formFields.forEach(field => {
      field.addEventListener('blur', validateField);
      field.addEventListener('input', clearFieldError);
    });

    // Form submission handler
    contactForm.addEventListener('submit', handleFormSubmit);

    // Disable submit button initially if form is invalid
    if (submitBtn) {
      submitBtn.disabled = !contactForm.checkValidity();
    }
  }

  /**
   * Validates a single form field
   * @param {Event} e - The blur event
   */
  function validateField(e) {
    const field = e.target;
    const fieldValue = field.value.trim();
    const fieldName = field.name || field.id;
    const errorElement = field.nextElementSibling;

    // Clear any existing error classes
    field.classList.remove('error');

    // Skip validation for select elements on blur
    if (field.tagName === 'SELECT') return;

    // Validate based on field type
    if (field.required && fieldValue === '') {
      showError(field, `${fieldName} is required`);
      return;
    }

    if (field.type === 'email' && !isValidEmail(fieldValue)) {
      showError(field, 'Please enter a valid email address');
      return;
    }

    if (field.id === 'message' && fieldValue.length < 10) {
      showError(field, 'Message should be at least 10 characters long');
      return;
    }
  }

  /**
   * Shows error message for a field
   * @param {HTMLElement} field - The form field
   * @param {string} message - The error message
   */
  function showError(field, message) {
    // Add error class to field
    field.classList.add('error');

    // Create or update error message element
    let errorElement = field.nextElementSibling;
    if (!errorElement || !errorElement.classList.contains('error-message')) {
      errorElement = document.createElement('div');
      errorElement.className = 'error-message';
      field.parentNode.insertBefore(errorElement, field.nextSibling);
    }

    errorElement.textContent = message;
    updateSubmitButtonState();
  }

  /**
   * Clears error state when user starts typing
   * @param {Event} e - The input event
   */
  function clearFieldError(e) {
    const field = e.target;
    field.classList.remove('error');

    const errorElement = field.nextElementSibling;
    if (errorElement && errorElement.classList.contains('error-message')) {
      errorElement.remove();
    }

    updateSubmitButtonState();
  }

  /**
   * Updates submit button state based on form validity
   */
  function updateSubmitButtonState() {
    if (submitBtn) {
      submitBtn.disabled = !contactForm.checkValidity();
    }
  }

  /**
   * Handles form submission
   * @param {Event} e - The submit event
   */
async function handleFormSubmit(e) {
  e.preventDefault();
  const form = e.target;
  const submitBtn = form.querySelector('button[type="submit"]');

  // Disable button + show loading
  submitBtn.disabled = true;
  submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Sending...';

  try {
    const response = await fetch(form.action, {
      method: 'POST',
      body: new FormData(form),
      headers: { 'Accept': 'application/json' }
    });

    if (response.ok) {
      showFormFeedback('success', 'Message sent! I’ll reply soon.');
      form.reset(); // Clear form
    } else {
      throw new Error('Failed to send');
    }
  } catch (error) {
    showFormFeedback('error', 'Oops! Could not send. Try emailing me directly at roubhizakarya@gmail.com');
  } finally {
    submitBtn.disabled = false;
    submitBtn.innerHTML = '<i class="fas fa-paper-plane"></i> Send Message';
  }
}

  /**
   * Shows success/error feedback message
   * @param {string} type - 'success' or 'error'
   * @param {string} message - The feedback message
   */
  function showFormFeedback(type, message) {
    // Remove any existing feedback messages
    const existingFeedback = document.querySelector('.form-feedback');
    if (existingFeedback) {
      existingFeedback.remove();
    }

    // Create feedback element
    const feedbackElement = document.createElement('div');
    feedbackElement.className = `form-feedback ${type}`;
    feedbackElement.innerHTML = `
      <i class="fas fa-${type === 'success' ? 'check-circle' : 'exclamation-circle'}"></i>
      <span>${message}</span>
    `;

    // Insert feedback before the form
    contactForm.parentNode.insertBefore(feedbackElement, contactForm);

    // Remove feedback after 5 seconds
    setTimeout(() => {
      feedbackElement.remove();
    }, 5000);
  }

  /**
   * Validates email format
   * @param {string} email - The email to validate
   * @returns {boolean} - True if email is valid
   */
  function isValidEmail(email) {
    const re = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return re.test(email);
  }

  /**
   * Gets form data as an object
   * @param {HTMLFormElement} form - The form element
   * @returns {Object} - Form data as key-value pairs
   */
  function getFormData(form) {
    const formData = new FormData(form);
    const data = {};
    formData.forEach((value, key) => {
      data[key] = value;
    });
    return data;
  }

  // GitHub link click handler
  const githubLink = document.querySelector('.github-link');
  if (githubLink) {
    githubLink.addEventListener('click', function(e) {
      e.preventDefault();
      // Replace with your actual GitHub repository URL
      window.open('https://github.com/zuikre/Datakarya', '_blank');
    });
  }
});

/**
 * script.js
 * Dynamic neural-network visualization
 * - Builds an SVG overlay connecting "neuron" elements
 * - Animates activations and connection pulses
 * - Play/Pause, Speed control, Mode switching
 * - Responsive & accessible (respects prefers-reduced-motion)
 *
 * Usage:
 * - ensure HTML uses .neural-network-visualization and .layer/.neuron structure
 * - Optional controls: #btn-play, #speed, #mode (script will bind if present)
 *
 * Notes:
 * - This code tries to be robust: works with varying neuron counts per layer
 * - Uses requestAnimationFrame for animation loop
 */

/* ---------------------------
   Config / internal state
   --------------------------- */
(function () {
  'use strict';

  const root = document.documentElement;
  const viz = document.getElementById('neuralViz') || document.querySelector('.neural-network-visualization');
  if (!viz) return;

  // find layers by data attribute or fallback to class names
  const layers = Array.from(viz.querySelectorAll('.layer'));
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // SVG overlay
  let svg;
  let connGroup;

  // animation state
  let lastTime = 0;
  let elapsed = 0;
  let running = true;
  let speed = 1; // multiplier (connected to --speed CSS var)
  let mode = 'default'; // 'default' | 'activation' | 'gradient' etc.

  // store connection objects for animation
  const connections = [];

  /* ---------------------------
     Utilities
     --------------------------- */

  function createSVG(tag, attrs = {}) {
    const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    for (const k in attrs) el.setAttribute(k, attrs[k]);
    return el;
  }

  // compute absolute center of an element (relative to viz container)
  function centerOf(el) {
    const rect = el.getBoundingClientRect();
    const pRect = viz.getBoundingClientRect();
    return {
      x: rect.left + rect.width / 2 - pRect.left,
      y: rect.top + rect.height / 2 - pRect.top,
    };
  }

  // compute animation-friendly dash offset speed
  function connectionSpeedFactor(base = 80) {
    return base * (1 / Math.max(0.25, Math.min(3, speed))); // inverse relation
  }

  /* ---------------------------
     Build SVG overlay and gradients
     --------------------------- */
  function ensureSVG() {
    // remove old
    const existing = viz.querySelector('svg.nn-svg');
    if (existing) existing.remove();

    const w = viz.clientWidth;
    const h = viz.clientHeight;
    svg = createSVG('svg', { class: 'nn-svg', width: w, height: h, viewBox: `0 0 ${w} ${h}`, preserveAspectRatio: 'none' });

    // defs (gradients)
    const defs = createSVG('defs');
    // gradient for connection pulse
    const grad = createSVG('linearGradient', { id: 'grad-conn', x1: '0', x2: '1' });
    grad.appendChild(createSVG('stop', { offset: '0%', 'stop-color': '#7c3aed', 'stop-opacity': '1' }));
    grad.appendChild(createSVG('stop', { offset: '60%', 'stop-color': '#06b6d4', 'stop-opacity': '1' }));
    defs.appendChild(grad);
    svg.appendChild(defs);

    connGroup = createSVG('g', { class: 'connections-group', 'aria-hidden': 'true' });
    svg.appendChild(connGroup);

    viz.appendChild(svg);
  }

  /* ---------------------------
     Build connections array
     --------------------------- */
  function buildConnections() {
    connections.length = 0;
    // find neuron elements per layer in DOM order
    const layerNodes = layers.map(l => Array.from(l.querySelectorAll('.neuron')));

    // create connections between consecutive layers
    for (let i = 0; i < layerNodes.length - 1; i++) {
      const fromNeurons = layerNodes[i];
      const toNeurons = layerNodes[i + 1];

      fromNeurons.forEach((fEl, fi) => {
        toNeurons.forEach((tEl, ti) => {
          const line = createSVG('path', { class: 'nn-conn' });
          // initial empty path
          line.setAttribute('d', 'M0 0 L0 0');
          line.setAttribute('fill', 'none');
          connGroup.appendChild(line);

          connections.push({
            from: fEl,
            to: tEl,
            path: line,
            phase: Math.random() * Math.PI * 2, // random phase for variation
            pulse: Math.random() > 0.9 ? 1 : 0, // some lines are marked for pulse
            speedOffset: Math.random() * 0.8 + 0.6,
          });
        });
      });
    }
  }

  /* ---------------------------
     Recalculate all geometry (on resize)
     --------------------------- */
  function layoutConnections() {
    if (!svg) return;
    const w = viz.clientWidth;
    const h = viz.clientHeight;
    svg.setAttribute('width', w);
    svg.setAttribute('height', h);
    svg.setAttribute('viewBox', `0 0 ${w} ${h}`);

    // update each connection path BETWEEN centers with a nice cubic curve
    connections.forEach(conn => {
      const a = centerOf(conn.from);
      const b = centerOf(conn.to);

      // create a smooth curve (control points at 25%/75% of X spread, offset by Y)
      const dx = b.x - a.x;
      const midX = a.x + dx * 0.45;
      const cp1y = a.y - Math.min(40, Math.abs(dx) * 0.16);
      const cp2y = b.y + Math.min(40, Math.abs(dx) * 0.16);

      const d = `M ${a.x.toFixed(1)} ${a.y.toFixed(1)} C ${midX.toFixed(1)} ${cp1y.toFixed(1)}, ${midX.toFixed(1)} ${cp2y.toFixed(1)}, ${b.x.toFixed(1)} ${b.y.toFixed(1)}`;
      conn.path.setAttribute('d', d);

      // set stroke attributes for animation
      if (conn.pulse) {
        conn.path.classList.add('pulse', 'animated');
      } else {
        conn.path.classList.remove('pulse');
      }

      // prepare length / dash for animation orb
      try {
        const L = conn.path.getTotalLength();
        conn.length = L;
        conn.path.style.strokeDasharray = `${Math.max(6, Math.round(L / 40))} ${Math.max(6, Math.round(L / 20))}`;
      } catch (e) {
        conn.length = 200;
      }
    });
  }

  /* ---------------------------
     Animation loop: animate connections (dash offset) and neurons activation
     --------------------------- */
  // simple activation pattern: oscillate neurons with per-node offsets
  function animateFrame(ts) {
    if (!running) {
      lastTime = ts;
      requestAnimationFrame(animateFrame);
      return;
    }

    if (!lastTime) lastTime = ts;
    const dt = (ts - lastTime) / 1000; // seconds
    lastTime = ts;
    elapsed += dt * speed;

    // update CSS variable for speed to allow transitions that rely on it
    root.style.setProperty('--speed', String(speed));

    // animate connections stroke-dashoffset for moving "orb" effect
    connections.forEach((conn, idx) => {
      const phase = (elapsed * connectionSpeedFactor(conn.speedOffset) + conn.phase);
      const dashOffset = (phase * 24) % (conn.length || 200);
      // invert or vary sign for visual interest
      conn.path.style.strokeDashoffset = `${-dashOffset.toFixed(2)}`;
    });

    // animate neuron activations: simple oscillation
    layers.forEach((layer, li) => {
      const neurons = Array.from(layer.querySelectorAll('.neuron'));
      neurons.forEach((n, ni) => {
        // compute pseudo-random activation based on elapsed time and index
        const base = Math.sin(elapsed * (0.8 + li * 0.2 + ni * 0.12) + li * 0.6 + ni * 0.3);
        const act = (base + 1) / 2; // 0..1
        const threshold = 0.65; // when to consider as 'active'
        if (!prefersReducedMotion) {
          if (act > threshold) {
            if (!n.classList.contains('active')) n.classList.add('active');
          } else {
            if (n.classList.contains('active')) n.classList.remove('active');
          }
          // subtle scale via inline style when active to make it responsive to speed
          const scale = 1 + (act > threshold ? (act - threshold) * 0.45 : (act - 0.45) * 0.12);
          n.style.transform = `scale(${scale.toFixed(3)})`;
        } else {
          // reduced motion -> keep minimal toggles
          n.classList.remove('active');
          n.style.transform = '';
        }
      });
    });

    requestAnimationFrame(animateFrame);
  }

  /* ---------------------------
     Controls (play/pause, speed, mode)
     --------------------------- */
  function bindControls() {
    const btnPlay = document.getElementById('btn-play');
    const speedEl = document.getElementById('speed');
    const modeEl = document.getElementById('mode');

    if (btnPlay) {
      btnPlay.addEventListener('click', () => {
        running = !running;
        btnPlay.textContent = running ? 'Pause' : 'Play';
        btnPlay.setAttribute('aria-pressed', String(!running));
      });
    }

    if (speedEl) {
      // initialize
      speed = parseFloat(speedEl.value) || 1;
      speedEl.addEventListener('input', (e) => {
        const v = parseFloat(e.target.value) || 1;
        speed = v;
      });
    }

    if (modeEl) {
      modeEl.addEventListener('change', (e) => {
        mode = e.target.value;
        // for now modes affect only style/class toggles; extend as desired
        viz.dataset.mode = mode;
      });
    }
  }

  /* ---------------------------
     Tilt / parallax on mouse move
     --------------------------- */
  function bindTilt() {
    if (prefersReducedMotion) return; // no tilt if user prefers reduced motion

    let rect = viz.getBoundingClientRect();
    let active = false;
    function onMove(e) {
      const x = (e.clientX || (e.touches && e.touches[0] && e.touches[0].clientX) || 0);
      const y = (e.clientY || (e.touches && e.touches[0] && e.touches[0].clientY) || 0);
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      const dx = (x - cx) / rect.width;
      const dy = (y - cy) / rect.height;
      const rx = (-dy * 8).toFixed(2);
      const ry = (dx * 8).toFixed(2);
      viz.style.transform = `perspective(1000px) rotateX(${rx}deg) rotateY(${ry}deg) translateZ(0)`;
      active = true;
    }
    function onLeave() {
      viz.style.transform = '';
      active = false;
    }
    viz.addEventListener('mousemove', onMove);
    viz.addEventListener('touchmove', onMove, { passive: true });
    viz.addEventListener('mouseleave', onLeave);
    window.addEventListener('resize', () => rect = viz.getBoundingClientRect());
  }

  /* ---------------------------
     Resize observer & initialization
     --------------------------- */
  function init() {
    // create SVG overlay and populate
    ensureSVG();
    buildConnections();
    layoutConnections();

    // bind controls
    bindControls();
    bindTilt();

    // start loop
    requestAnimationFrame(animateFrame);

    // responsive: recalc on resize
    let resizeId;
    window.addEventListener('resize', () => {
      clearTimeout(resizeId);
      resizeId = setTimeout(() => {
        layoutConnections();
      }, 120);
    });

    // Also observe DOM changes that could change neuron positions (# dynamic layer changes)
    const mo = new MutationObserver(() => {
      // rebuild connections if neuron counts change
      connGroup && (connGroup.innerHTML = '');
      buildConnections();
      layoutConnections();
    });
    mo.observe(viz, { childList: true, subtree: true });
  }

  // small delay to ensure CSS and layout ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    setTimeout(init, 40);
  }

  /* ---------------------------
     Accessibility helpers (ARIA)
     --------------------------- */
  // set roles for layers / neurons
  layers.forEach((layer, i) => {
    layer.setAttribute('role', 'group');
    layer.setAttribute('aria-label', layer.dataset.layer || `layer-${i+1}`);
    const neurons = layer.querySelectorAll('.neuron');
    neurons.forEach((n, ni) => {
      n.setAttribute('role', 'img');
      n.setAttribute('aria-label', (n.dataset.label || `node ${ni + 1}`));
      n.tabIndex = 0;
      // on focus, highlight surrounding connections for clarity
      n.addEventListener('focus', () => {
        // highlight connections that include this neuron
        connections.forEach(c => {
          if (c.from === n || c.to === n) c.path.classList.add('pulse');
        });
      });
      n.addEventListener('blur', () => {
        connections.forEach(c => {
          if (c.from === n || c.to === n) c.path.classList.remove('pulse');
        });
      });
    });
  });

})();

const input = document.querySelector('.search-input');

input.addEventListener('input', () => {
  if (input.value.length > 0) {
    input.style.boxShadow =
      "0 0 18px rgba(0,255,200,0.7), 0 0 28px rgba(157,78,221,0.8)";
  } else {
    input.style.boxShadow =
      "0 0 8px rgba(157,78,221,0.4), inset 0 0 4px rgba(255,255,255,0.05)";
  }
});