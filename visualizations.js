/**
 * Algorithm Visualizations
 * A cinematic, interactive visualization system for machine learning algorithms
 * with advanced animations, 3D visualizations, and comprehensive controls
 */

// =============================================
// Global Configuration and Helper Functions
// =============================================

// Enhanced color palette with more options
const COLORS = {
  primary: '#4e79a7',
  secondary: '#f28e2b',
  accent: '#e15759',
  background: '#f0f0f0',
  text: '#333333',
  positive: '#59a14f',
  negative: '#edc948',
  highlight: '#76b7b2',
  grid: '#dddddd',
  dark: '#2c3e50',
  light: '#ecf0f1',
  purple: '#9c755f',
  pink: '#e377c2',
  blue: '#17becf',
  green: '#2ca02c',
  red: '#d62728',
  gray: '#7f7f7f',
  spectrum: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
};

// Animation timing defaults with more options
// In your animation configuration:
const ANIMATION = {
  duration: 1000, // Reduced from 1500
  easing: 'easeOutQuad', // Faster ending
  frameRate: 30, // Reduced from 60 (still smooth)
  delay: 20, // Reduced between points
  batchSize: 10 // Render points in groups
};

// Math utility functions with enhancements
const MathUtils = {
  // Linear interpolation
  lerp: (a, b, t) => a + (b - a) * t,
  
  // Clamp value between min and max
  clamp: (value, min, max) => Math.min(Math.max(value, min), max),
  
  // Map value from one range to another
  map: (value, inMin, inMax, outMin, outMax) => 
    ((value - inMin) * (outMax - outMin)) / (inMax - inMin) + outMin,
  
  // Sigmoid function
  sigmoid: (x) => 1 / (1 + Math.exp(-x)),
  
  // Softmax function
  softmax: (arr) => {
    const max = Math.max(...arr);
    const exps = arr.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  },
  
  // Random number in range
  random: (min, max) => Math.random() * (max - min) + min,
  
  // Random integer in range
  randomInt: (min, max) => Math.floor(Math.random() * (max - min + 1)) + min,
  
  // Gaussian random
  gaussianRandom: (mean = 0, stdev = 1) => {
    const u = 1 - Math.random();
    const v = Math.random();
    const z = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    return z * stdev + mean;
  },
  
  // Distance between two points
  distance: (x1, y1, x2, y2) => Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2)),
  
  // Dot product of two vectors
  dot: (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0),
  
  // Magnitude of a vector
  magnitude: (v) => Math.sqrt(v.reduce((sum, val) => sum + val * val, 0)),
  
  // Normalize a vector
  normalize: (v) => {
    const mag = MathUtils.magnitude(v);
    return v.map(x => x / mag);
  },
  
  // Matrix multiplication
  matMul: (a, b) => {
    const result = [];
    for (let i = 0; i < a.length; i++) {
      result[i] = [];
      for (let j = 0; j < b[0].length; j++) {
        let sum = 0;
        for (let k = 0; k < a[0].length; k++) {
          sum += a[i][k] * b[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  },
  
  // Transpose matrix
  transpose: (matrix) => {
    return matrix[0].map((_, i) => matrix.map(row => row[i]));
  }
};

// Enhanced easing functions
const Easing = {
  linear: t => t,
  easeInQuad: t => t * t,
  easeOutQuad: t => t * (2 - t),
  easeInOutQuad: t => t < 0.5 ? 2 * t * t : -1 + (4 - 2 * t) * t,
  easeInCubic: t => t * t * t,
  easeOutCubic: t => (--t) * t * t + 1,
  easeInOutCubic: t => t < 0.5 ? 4 * t * t * t : (t - 1) * (2 * t - 2) * (2 * t - 2) + 1,
  easeInQuart: t => t * t * t * t,
  easeOutQuart: t => 1 - (--t) * t * t * t,
  easeInOutQuart: t => t < 0.5 ? 8 * t * t * t * t : 1 - 8 * (--t) * t * t * t,
  easeInQuint: t => t * t * t * t * t,
  easeOutQuint: t => 1 + (--t) * t * t * t * t,
  easeInOutQuint: t => t < 0.5 ? 16 * t * t * t * t * t : 1 + 16 * (--t) * t * t * t * t,
  easeInSine: t => 1 - Math.cos(t * Math.PI / 2),
  easeOutSine: t => Math.sin(t * Math.PI / 2),
  easeInOutSine: t => -(Math.cos(Math.PI * t) - 1) / 2,
  easeInExpo: t => t === 0 ? 0 : Math.pow(2, 10 * t - 10),
  easeOutExpo: t => t === 1 ? 1 : 1 - Math.pow(2, -10 * t),
  easeInOutExpo: t => {
    if (t === 0) return 0;
    if (t === 1) return 1;
    return t < 0.5 ? Math.pow(2, 20 * t - 10) / 2 : (2 - Math.pow(2, -20 * t + 10)) / 2;
  },
  easeInCirc: t => 1 - Math.sqrt(1 - t * t),
  easeOutCirc: t => Math.sqrt(1 - (t - 1) * (t - 1)),
  easeInOutCirc: t => t < 0.5 ? (1 - Math.sqrt(1 - 4 * t * t)) / 2 : (Math.sqrt(1 - (-2 * t + 2) * (-2 * t + 2)) + 1) / 2,
  easeInElastic: t => {
    const c4 = (2 * Math.PI) / 3;
    return t === 0 ? 0 : t === 1 ? 1 : -Math.pow(2, 10 * t - 10) * Math.sin((t * 10 - 10.75) * c4);
  },
  easeOutElastic: t => {
    const c4 = (2 * Math.PI) / 3;
    return t === 0 ? 0 : t === 1 ? 1 : Math.pow(2, -10 * t) * Math.sin((t * 10 - 0.75) * c4) + 1;
  },
  easeInOutElastic: t => {
    const c5 = (2 * Math.PI) / 4.5;
    return t === 0 ? 0 : t === 1 ? 1 : t < 0.5 ? 
      -(Math.pow(2, 20 * t - 10) * Math.sin((20 * t - 11.125) * c5)) / 2 : 
      (Math.pow(2, -20 * t + 10) * Math.sin((20 * t - 11.125) * c5)) / 2 + 1;
  },
  easeInBack: t => {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return c3 * t * t * t - c1 * t * t;
  },
  easeOutBack: t => {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
  },
  easeInOutBack: t => {
    const c1 = 1.70158;
    const c2 = c1 * 1.525;
    return t < 0.5 ? (Math.pow(2 * t, 2) * ((c2 + 1) * 2 * t - c2)) / 2 : 
      (Math.pow(2 * t - 2, 2) * ((c2 + 1) * (t * 2 - 2) + c2) + 2) / 2;
  },
  easeInBounce: t => 1 - Easing.easeOutBounce(1 - t),
  easeOutBounce: t => {
    const n1 = 7.5625;
    const d1 = 2.75;
    
    if (t < 1 / d1) {
      return n1 * t * t;
    } else if (t < 2 / d1) {
      return n1 * (t -= 1.5 / d1) * t + 0.75;
    } else if (t < 2.5 / d1) {
      return n1 * (t -= 2.25 / d1) * t + 0.9375;
    } else {
      return n1 * (t -= 2.625 / d1) * t + 0.984375;
    }
  },
  easeInOutBounce: t => t < 0.5 ? (1 - Easing.easeOutBounce(1 - 2 * t)) / 2 : (1 + Easing.easeOutBounce(2 * t - 1)) / 2
};

// Enhanced DOM and Canvas helper functions
const DomUtils = {
  // Create a canvas element with proper scaling for high DPI displays
  createCanvas: (containerId, width, height, options = {}) => {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error(`Container with ID ${containerId} not found`);
      return null;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    const canvas = document.createElement('canvas');
    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.style.backgroundColor = options.background || COLORS.background;
    canvas.style.borderRadius = options.borderRadius || '8px';
    canvas.style.boxShadow = options.boxShadow || '0 4px 6px rgba(0, 0, 0, 0.1)';
    
    const ctx = canvas.getContext('2d');
    ctx.scale(dpr, dpr);
    
    container.appendChild(canvas);
    return { canvas, ctx, width, height };
  },
  
  // Create SVG element with enhanced options
  createSVG: (containerId, width, height, options = {}) => {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error(`Container with ID ${containerId} not found`);
      return null;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('width', width);
    svg.setAttribute('height', height);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    svg.style.backgroundColor = options.background || COLORS.background;
    svg.style.borderRadius = options.borderRadius || '8px';
    svg.style.boxShadow = options.boxShadow || '0 4px 6px rgba(0, 0, 0, 0.1)';
    
    container.appendChild(svg);
    return svg;
  },

// Alternative WebGL implementation:
createWebGLPoints: function(gl, data) {
  const positions = new Float32Array(data.length * 2);
  const colors = new Float32Array(data.length * 3);
  
  data.forEach((point, i) => {
    positions[i*2] = toCanvasX(point.x);
    positions[i*2+1] = toCanvasY(point.y);
    
    const color = point.outlier 
      ? hexToRgb(COLORS.accent)
      : hexToRgb(COLORS.spectrum[point.label % COLORS.spectrum.length]);
    
    colors[i*3] = color.r;
    colors[i*3+1] = color.g;
    colors[i*3+2] = color.b;
  });

  // Create buffers and shaders here...
  // (WebGL implementation would continue)
},

  // Create WebGL context with Three.js
  createWebGL: (containerId, width, height, options = {}) => {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error(`Container with ID ${containerId} not found`);
      return null;
    }
    
    // Clear existing content
    container.innerHTML = '';
    
    // Check if Three.js is available
    if (typeof THREE === 'undefined') {
      console.error('Three.js is not loaded. Please include Three.js before using WebGL visualizations.');
      return null;
    }
    
    // Create renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.setClearColor(options.background || COLORS.background, 1);
    renderer.domElement.style.borderRadius = options.borderRadius || '8px';
    renderer.domElement.style.boxShadow = options.boxShadow || '0 4px 6px rgba(0, 0, 0, 0.1)';
    
    container.appendChild(renderer.domElement);
    
    // Create scene
    const scene = new THREE.Scene();
    
    // Create camera
    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 1000);
    camera.position.z = 5;
    
    return { renderer, scene, camera, container, width, height };
  },
  
  // Create controls panel with enhanced styling and functionality
  createControls: (containerId, controls, options = {}) => {
    let container;
    
    // Check if a specific controls container was provided
    if (options.controlsContainer) {
      container = document.getElementById(options.controlsContainer);
    }
    
    // Fallback to the main container
    if (!container) {
      container = document.getElementById(containerId);
    }
    
    if (!container) return null;
    
    // Clear existing controls if using external container
    if (options.controlsContainer) {
      container.innerHTML = '';
    }
    
    const panel = document.createElement('div');
    panel.className = 'controls-panel';
    panel.style.padding = '20px';
    panel.style.backgroundColor = '#f8f8f8';
    panel.style.borderRadius = '8px';
    panel.style.marginBottom = '20px';
    panel.style.boxShadow = '0 2px 10px rgba(0,0,0,0.1)';
    panel.style.fontFamily = 'Arial, sans-serif';
    
    // Add title for controls
    if (options.title) {
      const title = document.createElement('h3');
      title.textContent = options.title;
      title.style.margin = '0 0 15px 0';
      title.style.color = COLORS.dark;
      title.style.fontSize = '18px';
      title.style.fontWeight = '600';
      title.style.borderBottom = `2px solid ${COLORS.primary}`;
      title.style.paddingBottom = '8px';
      panel.appendChild(title);
    }
    
    // Add description if provided
    if (options.description) {
      const desc = document.createElement('p');
      desc.textContent = options.description;
      desc.style.margin = '0 0 15px 0';
      desc.style.color = COLORS.text;
      desc.style.fontSize = '14px';
      desc.style.lineHeight = '1.4';
      panel.appendChild(desc);
    }
    
    controls.forEach(control => {
      const controlDiv = document.createElement('div');
      controlDiv.style.marginBottom = '15px';
      controlDiv.style.display = 'flex';
      controlDiv.style.alignItems = 'center';
      controlDiv.style.gap = '12px';
      
      const label = document.createElement('label');
      label.textContent = control.label;
      label.style.minWidth = '150px';
      label.style.fontSize = '14px';
      label.style.color = '#555';
      label.style.fontWeight = '500';
      
      let input;
      
      if (control.type === 'range') {
        const inputContainer = document.createElement('div');
        inputContainer.style.flex = '1';
        inputContainer.style.display = 'flex';
        inputContainer.style.alignItems = 'center';
        inputContainer.style.gap = '10px';
        
        input = document.createElement('input');
        input.type = 'range';
        input.min = control.min;
        input.max = control.max;
        input.step = control.step || 0.1;
        input.value = control.value;
        input.style.flex = '1';
        input.style.height = '8px';
        input.style.borderRadius = '4px';
        input.style.background = '#ddd';
        input.style.outline = 'none';
        input.style.cursor = 'pointer';
        input.style.transition = 'all 0.2s';
        
        input.addEventListener('mouseover', () => {
          input.style.background = COLORS.primary + '80';
        });
        
        input.addEventListener('mouseout', () => {
          input.style.background = '#ddd';
        });
        
        const valueDisplay = document.createElement('span');
        valueDisplay.textContent = control.value;
        valueDisplay.style.minWidth = '50px';
        valueDisplay.style.fontSize = '14px';
        valueDisplay.style.fontWeight = '600';
        valueDisplay.style.color = COLORS.primary;
        valueDisplay.style.textAlign = 'right';
        
        if (control.onChange) {
          input.addEventListener('input', (e) => {
            const value = parseFloat(e.target.value);
            valueDisplay.textContent = control.format ? control.format(value) : value.toFixed(2);
            control.onChange(value);
          });
        }
        
        inputContainer.appendChild(input);
        inputContainer.appendChild(valueDisplay);
        
        controlDiv.appendChild(label);
        controlDiv.appendChild(inputContainer);
      }
      
      if (control.type === 'checkbox') {
        const checkboxContainer = document.createElement('div');
        checkboxContainer.style.display = 'flex';
        checkboxContainer.style.alignItems = 'center';
        checkboxContainer.style.gap = '8px';
        
        input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = control.checked || false;
        input.style.width = '18px';
        input.style.height = '18px';
        input.style.cursor = 'pointer';
        
        // Custom checkbox styling
        const customCheckbox = document.createElement('span');
        customCheckbox.style.display = 'inline-block';
        customCheckbox.style.width = '18px';
        customCheckbox.style.height = '18px';
        customCheckbox.style.borderRadius = '4px';
        customCheckbox.style.border = `2px solid ${COLORS.primary}`;
        customCheckbox.style.position = 'relative';
        customCheckbox.style.transition = 'all 0.2s';
        
        // Checkmark
        const checkmark = document.createElement('span');
        checkmark.style.position = 'absolute';
        checkmark.style.top = '2px';
        checkmark.style.left = '6px';
        checkmark.style.width = '5px';
        checkmark.style.height = '10px';
        checkmark.style.border = `solid ${COLORS.primary}`;
        checkmark.style.borderWidth = '0 2px 2px 0';
        checkmark.style.transform = 'rotate(45deg)';
        checkmark.style.opacity = input.checked ? '1' : '0';
        checkmark.style.transition = 'opacity 0.2s';
        customCheckbox.appendChild(checkmark);
        
        // Update checkmark when checkbox changes
        input.addEventListener('change', (e) => {
          checkmark.style.opacity = e.target.checked ? '1' : '0';
          if (control.onChange) control.onChange(e.target.checked);
        });
        
        checkboxContainer.appendChild(input);
        checkboxContainer.appendChild(customCheckbox);
        checkboxContainer.appendChild(label);
        
        controlDiv.appendChild(checkboxContainer);
      }
      
      if (control.type === 'select') {
        const selectContainer = document.createElement('div');
        selectContainer.style.flex = '1';
        selectContainer.style.position = 'relative';
        
        input = document.createElement('select');
        input.style.width = '100%';
        input.style.padding = '8px 12px';
        input.style.borderRadius = '4px';
        input.style.border = `1px solid ${COLORS.grid}`;
        input.style.backgroundColor = 'white';
        input.style.fontSize = '14px';
        input.style.color = COLORS.text;
        input.style.cursor = 'pointer';
        input.style.appearance = 'none';
        input.style.backgroundImage = 'url("data:image/svg+xml;charset=US-ASCII,%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%20width%3D%22292.4%22%20height%3D%22292.4%22%3E%3Cpath%20fill%3D%22%234e79a7%22%20d%3D%22M287%2069.4a17.6%2017.6%200%200%200-13-5.4H18.4c-5%200-9.3%201.8-12.9%205.4A17.6%2017.6%200%200%200%200%2082.2c0%205%201.8%209.3%205.4%2012.9l128%20127.9c3.6%203.6%207.8%205.4%2012.8%205.4s9.2-1.8%2012.8-5.4L287%2095c3.5-3.5%205.4-7.8%205.4-12.8%200-5-1.9-9.2-5.5-12.8z%22%2F%3E%3C%2Fsvg%3E")';
        input.style.backgroundRepeat = 'no-repeat';
        input.style.backgroundPosition = 'right 10px center';
        input.style.backgroundSize = '10px auto';
        
        control.options.forEach(option => {
          const opt = document.createElement('option');
          opt.value = option.value;
          opt.textContent = option.label;
          if (option.selected) opt.selected = true;
          input.appendChild(opt);
        });
        
        if (control.onChange) {
          input.addEventListener('change', (e) => control.onChange(e.target.value));
        }
        
        selectContainer.appendChild(input);
        controlDiv.appendChild(label);
        controlDiv.appendChild(selectContainer);
      }
      
      if (control.type === 'button') {
        input = document.createElement('button');
        input.textContent = control.label;
        input.style.padding = '8px 16px';
        input.style.borderRadius = '4px';
        input.style.border = 'none';
        input.style.backgroundColor = COLORS.primary;
        input.style.color = 'white';
        input.style.fontSize = '14px';
        input.style.fontWeight = '500';
        input.style.cursor = 'pointer';
        input.style.transition = 'all 0.2s';
        input.style.flex = '1';
        
        input.addEventListener('mouseover', () => {
          input.style.backgroundColor = COLORS.secondary;
        });
        
        input.addEventListener('mouseout', () => {
          input.style.backgroundColor = COLORS.primary;
        });
        
        if (control.onClick) {
          input.addEventListener('click', control.onClick);
        }
        
        controlDiv.appendChild(input);
      }
      
      if (control.type === 'color') {
        const colorContainer = document.createElement('div');
        colorContainer.style.display = 'flex';
        colorContainer.style.alignItems = 'center';
        colorContainer.style.gap = '10px';
        
        input = document.createElement('input');
        input.type = 'color';
        input.value = control.value || COLORS.primary;
        input.style.width = '40px';
        input.style.height = '30px';
        input.style.cursor = 'pointer';
        
        const valueDisplay = document.createElement('span');
        valueDisplay.textContent = input.value;
        valueDisplay.style.fontSize = '14px';
        valueDisplay.style.color = COLORS.text;
        
        if (control.onChange) {
          input.addEventListener('input', (e) => {
            valueDisplay.textContent = e.target.value;
            control.onChange(e.target.value);
          });
        }
        
        colorContainer.appendChild(label);
        colorContainer.appendChild(input);
        colorContainer.appendChild(valueDisplay);
        
        controlDiv.appendChild(colorContainer);
      }
      
      if (control.type === 'text') {
        input = document.createElement('input');
        input.type = 'text';
        input.value = control.value || '';
        input.style.flex = '1';
        input.style.padding = '8px 12px';
        input.style.borderRadius = '4px';
        input.style.border = `1px solid ${COLORS.grid}`;
        input.style.fontSize = '14px';
        
        if (control.onChange) {
          input.addEventListener('input', (e) => control.onChange(e.target.value));
        }
        
        controlDiv.appendChild(label);
        controlDiv.appendChild(input);
      }
      
      if (control.type === 'radio-group') {
        const radioContainer = document.createElement('div');
        radioContainer.style.display = 'flex';
        radioContainer.style.flexDirection = 'column';
        radioContainer.style.gap = '8px';
        
        const groupLabel = document.createElement('div');
        groupLabel.textContent = control.label;
        groupLabel.style.fontSize = '14px';
        groupLabel.style.color = '#555';
        groupLabel.style.fontWeight = '500';
        groupLabel.style.marginBottom = '5px';
        radioContainer.appendChild(groupLabel);
        
        control.options.forEach((option, i) => {
          const optionContainer = document.createElement('div');
          optionContainer.style.display = 'flex';
          optionContainer.style.alignItems = 'center';
          optionContainer.style.gap = '8px';
          
          const radio = document.createElement('input');
          radio.type = 'radio';
          radio.name = control.name || 'radio-group';
          radio.value = option.value;
          radio.id = `${control.name || 'radio'}-${i}`;
          radio.checked = option.selected || false;
          radio.style.width = '16px';
          radio.style.height = '16px';
          radio.style.cursor = 'pointer';
          
          const radioLabel = document.createElement('label');
          radioLabel.htmlFor = radio.id;
          radioLabel.textContent = option.label;
          radioLabel.style.fontSize = '14px';
          radioLabel.style.color = COLORS.text;
          radioLabel.style.cursor = 'pointer';
          
          if (control.onChange) {
            radio.addEventListener('change', (e) => {
              if (e.target.checked) control.onChange(e.target.value);
            });
          }
          
          optionContainer.appendChild(radio);
          optionContainer.appendChild(radioLabel);
          radioContainer.appendChild(optionContainer);
        });
        
        controlDiv.appendChild(radioContainer);
      }
      
      panel.appendChild(controlDiv);
    });
    
    // Add the panel to the container
    if (options.controlsContainer) {
      container.appendChild(panel);
    } else {
      container.insertBefore(panel, container.firstChild);
    }
    
    return panel;
  },
  
  // Create a tabbed interface for multiple visualizations
  createTabs: (containerId, tabs, options = {}) => {
    const container = document.getElementById(containerId);
    if (!container) return null;
    
    // Clear existing content
    container.innerHTML = '';
    
    // Create tab container
    const tabContainer = document.createElement('div');
    tabContainer.style.display = 'flex';
    tabContainer.style.marginBottom = '10px';
    tabContainer.style.borderBottom = `2px solid ${COLORS.grid}`;
    
    // Create content container
    const contentContainer = document.createElement('div');
    contentContainer.style.padding = '15px';
    contentContainer.style.backgroundColor = options.background || COLORS.background;
    contentContainer.style.borderRadius = '0 0 8px 8px';
    
    // Create tabs
    tabs.forEach((tab, index) => {
      const tabElement = document.createElement('button');
      tabElement.textContent = tab.label;
      tabElement.style.padding = '8px 16px';
      tabElement.style.border = 'none';
      tabElement.style.backgroundColor = 'transparent';
      tabElement.style.color = index === 0 ? COLORS.primary : COLORS.text;
      tabElement.style.fontSize = '14px';
      tabElement.style.fontWeight = '500';
      tabElement.style.cursor = 'pointer';
      tabElement.style.position = 'relative';
      tabElement.style.marginRight = '5px';
      tabElement.style.borderRadius = '4px 4px 0 0';
      tabElement.style.transition = 'all 0.2s';
      
      if (index === 0) {
        tabElement.style.backgroundColor = COLORS.background;
        tabElement.style.borderBottom = `2px solid ${COLORS.primary}`;
      }
      
      tabElement.addEventListener('click', () => {
        // Update all tabs
        tabs.forEach((_, i) => {
          const tabBtn = tabContainer.children[i];
          tabBtn.style.color = i === index ? COLORS.primary : COLORS.text;
          tabBtn.style.backgroundColor = i === index ? COLORS.background : 'transparent';
          tabBtn.style.borderBottom = i === index ? `2px solid ${COLORS.primary}` : 'none';
        });
        
        // Update content
        contentContainer.innerHTML = '';
        if (tab.content) {
          if (typeof tab.content === 'function') {
            tab.content(contentContainer);
          } else {
            contentContainer.appendChild(tab.content);
          }
        }
      });
      
      tabContainer.appendChild(tabElement);
    });
    
    // Add initial content
    if (tabs.length > 0 && tabs[0].content) {
      if (typeof tabs[0].content === 'function') {
        tabs[0].content(contentContainer);
      } else {
        contentContainer.appendChild(tabs[0].content);
      }
    }
    
    container.appendChild(tabContainer);
    container.appendChild(contentContainer);
    
    return { tabContainer, contentContainer };
  },
  
  // Create a tooltip element
  createTooltip: (text) => {
    const tooltip = document.createElement('div');
    tooltip.textContent = text;
    tooltip.style.position = 'absolute';
    tooltip.style.backgroundColor = 'rgba(0, 0, 0, 0.8)';
    tooltip.style.color = 'white';
    tooltip.style.padding = '5px 10px';
    tooltip.style.borderRadius = '4px';
    tooltip.style.fontSize = '12px';
    tooltip.style.pointerEvents = 'none';
    tooltip.style.zIndex = '1000';
    tooltip.style.transform = 'translate(-50%, -100%)';
    tooltip.style.opacity = '0';
    tooltip.style.transition = 'opacity 0.2s';
    
    document.body.appendChild(tooltip);
    
    return {
      element: tooltip,
      show: (x, y) => {
        tooltip.style.left = `${x}px`;
        tooltip.style.top = `${y}px`;
        tooltip.style.opacity = '1';
      },
      hide: () => {
        tooltip.style.opacity = '0';
      },
      destroy: () => {
        document.body.removeChild(tooltip);
      }
    };
  },
  
  // Create a legend for visualizations
  createLegend: (items, options = {}) => {
    const legend = document.createElement('div');
    legend.style.display = 'flex';
    legend.style.flexDirection = options.direction === 'horizontal' ? 'row' : 'column';
    legend.style.gap = '10px';
    legend.style.padding = '10px';
    legend.style.backgroundColor = options.background || 'rgba(255, 255, 255, 0.8)';
    legend.style.borderRadius = '4px';
    legend.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
    legend.style.position = options.position || 'absolute';
    
    if (options.position === 'absolute') {
      legend.style.top = options.top || '10px';
      legend.style.right = options.right || '10px';
      legend.style.left = options.left || 'auto';
      legend.style.bottom = options.bottom || 'auto';
    }
    
    items.forEach(item => {
      const itemContainer = document.createElement('div');
      itemContainer.style.display = 'flex';
      itemContainer.style.alignItems = 'center';
      itemContainer.style.gap = '6px';
      
      const colorBox = document.createElement('div');
      colorBox.style.width = '15px';
      colorBox.style.height = '15px';
      colorBox.style.backgroundColor = item.color;
      colorBox.style.borderRadius = '3px';
      colorBox.style.border = item.border ? `1px solid ${item.border}` : 'none';
      
      const label = document.createElement('span');
      label.textContent = item.label;
      label.style.fontSize = '12px';
      label.style.color = COLORS.text;
      
      itemContainer.appendChild(colorBox);
      itemContainer.appendChild(label);
      legend.appendChild(itemContainer);
    });
    
    return legend;
  }
};

// Enhanced data simulation functions
const DataSimulator = {
  // Generate linear regression data with more options
  generateLinearData: (params = {}) => {
    const {
      n_samples = 100,
      slope = 2,
      intercept = 1,
      noise = 0.5,
      x_min = 0,
      x_max = 10,
      outliers = 0,
      trend = 'linear' // 'linear', 'quadratic', 'exponential', 'logarithmic', 'sinusoidal'
    } = params;
    
    const data = [];
    
    // Generate main data points
    for (let i = 0; i < n_samples; i++) {
      const x = MathUtils.random(x_min, x_max);
      let y;
      
      switch (trend) {
        case 'quadratic':
          y = intercept + slope * x * x + MathUtils.gaussianRandom(0, noise);
          break;
        case 'exponential':
          y = intercept + Math.exp(slope * x) + MathUtils.gaussianRandom(0, noise);
          break;
        case 'logarithmic':
          y = intercept + slope * Math.log(x + 1) + MathUtils.gaussianRandom(0, noise);
          break;
        case 'sinusoidal':
          y = intercept + slope * Math.sin(x) + MathUtils.gaussianRandom(0, noise);
          break;
        default: // linear
          y = intercept + slope * x + MathUtils.gaussianRandom(0, noise);
      }
      
      data.push({ x, y });
    }
    
    // Add outliers if specified
    for (let i = 0; i < outliers; i++) {
      const x = MathUtils.random(x_min, x_max);
      const y = intercept + slope * x + MathUtils.gaussianRandom(0, noise * 5);
      data.push({ x, y, outlier: true });
    }
    
    return data;
  },
  
  // Generate logistic regression data with more options
  generateLogisticData: (params = {}) => {
    const {
      n_samples = 100,
      class_separation = 1.5,
      noise = 0.3,
      x_min = -5,
      x_max = 5,
      n_classes = 2,
      distribution = 'linear' // 'linear', 'circular', 'xor'
    } = params;
    
    const data = [];
    
    if (distribution === 'circular') {
      // Circular decision boundary
      const radius = (x_max - x_min) / 3;
      const centerX = (x_min + x_max) / 2;
      const centerY = (x_min + x_max) / 2;
      
      for (let i = 0; i < n_samples; i++) {
        const x = MathUtils.random(x_min, x_max);
        const y = MathUtils.random(x_min, x_max);
        
        const distance = Math.sqrt(Math.pow(x - centerX, 2) + Math.pow(y - centerY, 2));
        const decision = (distance - radius) + MathUtils.gaussianRandom(0, noise);
        const label = decision > 0 ? 1 : 0;
        
        data.push({ x, y, label });
      }
    } else if (distribution === 'xor') {
      // XOR-like pattern
      for (let i = 0; i < n_samples; i++) {
        const x = MathUtils.random(x_min, x_max);
        const y = MathUtils.random(x_min, x_max);
        
        const decision = (x * y > 0 ? 1 : -1) + MathUtils.gaussianRandom(0, noise);
        const label = decision > 0 ? 1 : 0;
        
        data.push({ x, y, label });
      }
    } else {
      // Linear decision boundary (default)
      for (let i = 0; i < n_samples; i++) {
        const x = MathUtils.random(x_min, x_max);
        const y = MathUtils.random(x_min, x_max);
        
        // Simple linear decision boundary with some noise
        const decision = class_separation * (x + y) + MathUtils.gaussianRandom(0, noise);
        const label = decision > 0 ? 1 : 0;
        
        data.push({ x, y, label });
      }
    }
    
    return data;
  },
  
  // Generate decision tree data with more options
  generateTreeData: (params = {}) => {
    const {
      n_samples = 200,
      n_classes = 3,
      x_min = 0,
      x_max = 10,
      y_min = 0,
      y_max = 10,
      distribution = 'concentric' // 'concentric', 'clusters', 'checkerboard'
    } = params;
    
    const data = [];
    
    if (distribution === 'clusters') {
      // Generate data in distinct clusters
      const clusterCenters = [];
      for (let i = 0; i < n_classes; i++) {
        clusterCenters.push({
          x: MathUtils.random(x_min + 1, x_max - 1),
          y: MathUtils.random(y_min + 1, y_max - 1),
          label: i
        });
      }
      
      for (let i = 0; i < n_samples; i++) {
        const cluster = clusterCenters[MathUtils.randomInt(0, n_classes - 1)];
        const x = MathUtils.clamp(MathUtils.gaussianRandom(cluster.x, 0.5), x_min, x_max);
        const y = MathUtils.clamp(MathUtils.gaussianRandom(cluster.y, 0.5), y_min, y_max);
        data.push({ x, y, label: cluster.label });
      }
    } else if (distribution === 'checkerboard') {
      // Checkerboard pattern
      const xStep = (x_max - x_min) / Math.ceil(Math.sqrt(n_classes));
      const yStep = (y_max - y_min) / Math.ceil(Math.sqrt(n_classes));
      
      for (let i = 0; i < n_samples; i++) {
        const x = MathUtils.random(x_min, x_max);
        const y = MathUtils.random(y_min, y_max);
        
        const xIdx = Math.floor((x - x_min) / xStep);
        const yIdx = Math.floor((y - y_min) / yStep);
        const label = (xIdx + yIdx) % n_classes;
        
        data.push({ x, y, label });
      }
    } else {
      // Create concentric circular classes (default)
      for (let i = 0; i < n_samples; i++) {
        const x = MathUtils.random(x_min, x_max);
        const y = MathUtils.random(y_min, y_max);
        
        // Distance from center
        const cx = (x_max - x_min) / 2;
        const cy = (y_max - y_min) / 2;
        const dist = Math.sqrt(Math.pow(x - cx, 2) + Math.pow(y - cy, 2));
        
        // Assign class based on distance bands
        const max_dist = Math.sqrt(Math.pow(cx, 2) + Math.pow(cy, 2));
        const class_width = max_dist / n_classes;
        const label = Math.min(Math.floor(dist / class_width), n_classes - 1);
        
        data.push({ x, y, label });
      }
    }
    
    return data;
  },
  
  // Generate clustering data
  generateClusteringData: (params = {}) => {
    const {
      n_samples = 200,
      n_clusters = 3,
      x_min = 0,
      x_max = 10,
      y_min = 0,
      y_max = 10,
      cluster_std = 0.5,
      distribution = 'blobs' // 'blobs', 'moons', 'circles'
    } = params;
    
    const data = [];
    
    if (distribution === 'moons') {
      // Two interleaving half circles
      const n_samples_per_class = Math.ceil(n_samples / 2);
      
      // First moon
      for (let i = 0; i < n_samples_per_class; i++) {
        const angle = Math.PI * Math.random();
        const distance = MathUtils.random(0.8, 1.2);
        const x = Math.cos(angle) * distance + 2;
        const y = Math.sin(angle) * distance;
        
        data.push({ 
          x: MathUtils.map(x, -2, 4, x_min, x_max),
          y: MathUtils.map(y, -2, 2, y_min, y_max),
          label: 0,
          trueCluster: 0
        });
      }
      
      // Second moon
      for (let i = 0; i < n_samples_per_class; i++) {
        const angle = Math.PI * Math.random();
        const distance = MathUtils.random(0.8, 1.2);
        const x = Math.cos(angle) * distance + 3;
        const y = -Math.sin(angle) * distance - 0.5;
        
        data.push({ 
          x: MathUtils.map(x, -2, 4, x_min, x_max),
          y: MathUtils.map(y, -2, 2, y_min, y_max),
          label: 1,
          trueCluster: 1
        });
      }
    } else if (distribution === 'circles') {
      // One circle inside another
      const n_samples_per_class = Math.ceil(n_samples / 2);
      
      // Inner circle
      for (let i = 0; i < n_samples_per_class; i++) {
        const angle = 2 * Math.PI * Math.random();
        const distance = MathUtils.random(0, 0.5);
        const x = Math.cos(angle) * distance + 2;
        const y = Math.sin(angle) * distance + 2;
        
        data.push({ 
          x: MathUtils.map(x, 0, 4, x_min, x_max),
          y: MathUtils.map(y, 0, 4, y_min, y_max),
          label: 0,
          trueCluster: 0
        });
      }
      
      // Outer circle
      for (let i = 0; i < n_samples_per_class; i++) {
        const angle = 2 * Math.PI * Math.random();
        const distance = MathUtils.random(1, 1.5);
        const x = Math.cos(angle) * distance + 2;
        const y = Math.sin(angle) * distance + 2;
        
        data.push({ 
          x: MathUtils.map(x, 0, 4, x_min, x_max),
          y: MathUtils.map(y, 0, 4, y_min, y_max),
          label: 1,
          trueCluster: 1
        });
      }
    } else {
      // Gaussian blobs (default)
      const clusterCenters = [];
      for (let i = 0; i < n_clusters; i++) {
        clusterCenters.push({
          x: MathUtils.random(x_min + 1, x_max - 1),
          y: MathUtils.random(y_min + 1, y_max - 1),
          label: i
        });
      }
      
      for (let i = 0; i < n_samples; i++) {
        const cluster = clusterCenters[MathUtils.randomInt(0, n_clusters - 1)];
        const x = MathUtils.clamp(MathUtils.gaussianRandom(cluster.x, cluster_std), x_min, x_max);
        const y = MathUtils.clamp(MathUtils.gaussianRandom(cluster.y, cluster_std), y_min, y_max);
        data.push({ x, y, label: cluster.label, trueCluster: cluster.label });
      }
    }
    
    return data;
  },
  
  // Generate neural network training data
  generateNeuralNetworkData: (params = {}) => {
    const {
      n_samples = 200,
      n_classes = 3,
      x_min = 0,
      x_max = 10,
      y_min = 0,
      y_max = 10,
      complexity = 1 // 1-5, higher means more complex decision boundaries
    } = params;
    
    const data = [];
    
    // Generate a complex decision boundary based on the complexity parameter
    for (let i = 0; i < n_samples; i++) {
      const x = MathUtils.random(x_min, x_max);
      const y = MathUtils.random(y_min, y_max);
      
      // Create a complex decision boundary that gets more complex with higher complexity
      let decision = 0;
      for (let c = 1; c <= complexity; c++) {
        decision += Math.sin(x * c * 0.5) * Math.cos(y * c * 0.3) * (1 / c);
      }
      
      // Add some noise
      decision += MathUtils.gaussianRandom(0, 0.1);
      
      // Assign class based on decision value
      const label = Math.floor(MathUtils.map(decision, -1, 1, 0, n_classes - 1));
      
      data.push({ x, y, label });
    }
    
    return data;
  },
  
  // Generate time series data
  generateTimeSeriesData: (params = {}) => {
    const {
      n_points = 100,
      trend = 0.1, // slope of the trend line
      seasonality = 0.5, // strength of seasonal pattern
      noise = 0.2, // amount of noise
      seasonality_period = 20 // how many points per season
    } = params;
    
    const data = [];
    let y = 0;
    
    for (let i = 0; i < n_points; i++) {
      // Trend component
      y += trend;
      
      // Seasonality component
      const seasonal = seasonality * Math.sin(2 * Math.PI * i / seasonality_period);
      
      // Noise component
      const randomNoise = noise * MathUtils.gaussianRandom(0, 1);
      
      // Combine components
      const value = y + seasonal + randomNoise;
      
      data.push({ x: i, y: value });
    }
    
    return data;
  },
  
  // Generate high-dimensional data (for PCA)
  generateHighDimData: (params = {}) => {
    const {
      n_samples = 100,
      n_features = 5,
      n_informative = 2,
      noise = 0.1,
      cluster_std = 1.0,
      n_clusters = 3
    } = params;
    
    const data = [];
    
    // Generate cluster centers in informative dimensions
    const centers = [];
    for (let i = 0; i < n_clusters; i++) {
      const center = [];
      for (let j = 0; j < n_informative; j++) {
        center.push(MathUtils.random(-5, 5));
      }
      centers.push(center);
    }
    
    // Generate samples
    for (let i = 0; i < n_samples; i++) {
      const clusterIdx = MathUtils.randomInt(0, n_clusters - 1);
      const point = [];
      
      // Informative dimensions
      for (let j = 0; j < n_informative; j++) {
        point.push(MathUtils.gaussianRandom(centers[clusterIdx][j], cluster_std));
      }
      
      // Non-informative dimensions (noise)
      for (let j = n_informative; j < n_features; j++) {
        point.push(MathUtils.gaussianRandom(0, noise));
      }
      
      data.push({
        values: point,
        cluster: clusterIdx,
        label: clusterIdx // For compatibility with other visualizations
      });
    }
    
    return data;
  }
};

// Enhanced animation system with timeline and sequencing
const AnimationSystem = {
  // Animate a value over time
  animateValue: (options) => {
    const {
      duration = 1000,
      easing = 'easeInOutQuad',
      onUpdate,
      onComplete,
      delay = 0
    } = options;
    
    let startTime;
    let animationId;
    let isComplete = false;
    
    const animate = (currentTime) => {
      if (!startTime) startTime = currentTime;
      const elapsed = currentTime - startTime - delay;
      
      if (elapsed < 0) {
        // Still in delay period
        animationId = requestAnimationFrame(animate);
        return;
      }
      
      const progress = Math.min(elapsed / duration, 1);
      const easedProgress = Easing[easing] ? Easing[easing](progress) : progress;
      
      onUpdate(easedProgress);
      
      if (progress < 1) {
        animationId = requestAnimationFrame(animate);
      } else if (!isComplete && onComplete) {
        isComplete = true;
        onComplete();
      }
    };
    
    animationId = requestAnimationFrame(animate);
    
    return {
      stop: () => {
        cancelAnimationFrame(animationId);
        if (!isComplete && onComplete) {
          isComplete = true;
          onComplete(true); // Pass true to indicate early stop
        }
      },
      isComplete: () => isComplete
    };
  },
  
  // Animate multiple values in parallel
  animateValues: (animations) => {
    const controllers = animations.map(anim => AnimationSystem.animateValue(anim));
    return {
      stop: () => controllers.forEach(ctrl => ctrl.stop()),
      isComplete: () => controllers.every(ctrl => ctrl.isComplete())
    };
  },
  
  // Animate values in sequence
  animateSequence: (animations) => {
    let currentIndex = 0;
    let currentController = null;
    let isComplete = false;
    
    const startNext = () => {
      if (currentIndex >= animations.length) {
        isComplete = true;
        if (animations.onComplete) animations.onComplete();
        return;
      }
      
      const currentAnim = animations[currentIndex];
      currentController = AnimationSystem.animateValue({
        ...currentAnim,
        onComplete: () => {
          currentIndex++;
          startNext();
        }
      });
    };
    
    startNext();
    
    return {
      stop: () => {
        if (currentController) currentController.stop();
        isComplete = true;
      },
      isComplete: () => isComplete
    };
  },
  
  // Create a timeline for complex animations
  createTimeline: () => {
    const animations = [];
    let isPlaying = false;
    let controllers = [];
    
    return {
      add: (animation, options = {}) => {
        animations.push({
          animation,
          delay: options.delay || 0,
          at: options.at || null // absolute time
        });
      },
      
      play: (onComplete) => {
        if (isPlaying) return;
        isPlaying = true;
        
        // Calculate start times based on delays and absolute times
        let lastEndTime = 0;
        const timeline = animations.map(item => {
          let startTime;
          
          if (item.at !== null) {
            startTime = item.at;
          } else {
            startTime = lastEndTime + item.delay;
          }
          
          lastEndTime = startTime + (item.animation.duration || 1000);
          
          return {
            ...item.animation,
            delay: startTime
          };
        });
        
        controllers = AnimationSystem.animateSequence({
          ...timeline,
          onComplete: () => {
            isPlaying = false;
            if (onComplete) onComplete();
          }
        });
      },
      
      stop: () => {
        if (controllers) controllers.stop();
        isPlaying = false;
      },
      
      isPlaying: () => isPlaying
    };
  },
  
  // Animate elements along a path
  animateAlongPath: (pathPoints, options) => {
    const {
      duration = 1000,
      easing = 'linear',
      onUpdate,
      onComplete
    } = options;
    
    // Calculate cumulative distances for each segment
    const segmentDistances = [];
    let totalDistance = 0;
    
    for (let i = 1; i < pathPoints.length; i++) {
      const dist = MathUtils.distance(
        pathPoints[i-1].x, pathPoints[i-1].y,
        pathPoints[i].x, pathPoints[i].y
      );
      segmentDistances.push(dist);
      totalDistance += dist;
    }
    
    return AnimationSystem.animateValue({
      duration,
      easing,
      onUpdate: (progress) => {
        const targetDist = progress * totalDistance;
        let accumulatedDist = 0;
        let segmentIndex = 0;
        
        // Find which segment we're in
        while (segmentIndex < segmentDistances.length && 
               accumulatedDist + segmentDistances[segmentIndex] < targetDist) {
          accumulatedDist += segmentDistances[segmentIndex];
          segmentIndex++;
        }
        
        if (segmentIndex >= segmentDistances.length) {
          // At the end of the path
          const lastPoint = pathPoints[pathPoints.length - 1];
          onUpdate(lastPoint.x, lastPoint.y, progress);
          return;
        }
        
        // Calculate position within the current segment
        const segmentProgress = (targetDist - accumulatedDist) / segmentDistances[segmentIndex];
        const startPoint = pathPoints[segmentIndex];
        const endPoint = pathPoints[segmentIndex + 1];
        
        const x = MathUtils.lerp(startPoint.x, endPoint.x, segmentProgress);
        const y = MathUtils.lerp(startPoint.y, endPoint.y, segmentProgress);
        
        onUpdate(x, y, progress);
      },
      onComplete
    });
  }
};

// Modified point appearance animation:
function animatePointsAppearance() {
  const batchSize = ANIMATION.batchSize;
  const totalBatches = Math.ceil(data.length / batchSize);
  
  AnimationSystem.animateValue({
    duration: ANIMATION.duration * 0.3, // Shorter phase for points
    easing: ANIMATION.easing,
    onUpdate: (progress) => {
      const visiblePoints = Math.floor(progress * data.length);
      
      ctx.clearRect(0, 0, width, height);
      drawGrid();
      drawDataPoints(data.slice(0, visiblePoints), ctx, toCanvasX, toCanvasY);
      
      // Pulse effect for newly appearing points
      if (visiblePoints > 0) {
        const newestPoint = data[visiblePoints - 1];
        ctx.save();
        ctx.beginPath();
        ctx.arc(
          toCanvasX(newestPoint.x),
          toCanvasY(newestPoint.y),
          10 * (1 - (progress % 0.1) * 10), // Pulsing effect
          0, 
          Math.PI * 2
        );
        ctx.strokeStyle = COLORS.highlight;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
      }
    }
  });
}

// Automatically adjust rendering based on dataset size
function getOptimizedParams(dataLength) {
  return {
    pointRadius: dataLength > 500 ? 3 : 5,
    batchSize: Math.max(5, Math.floor(dataLength / 100)),
    frameRate: dataLength > 300 ? 24 : 60
  };
}

// =============================================
// Enhanced Linear Regression Visualizations
// =============================================

function visualizeLinearRegression(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 100,
    noise: 0.5,
    slope: 2,
    intercept: 1,
    show_residuals: true,
    show_confidence: false,
    show_prediction: false,
    animation_duration: 1500,
    interactive: true,
    controlsContainer: null,
    trend: 'linear', // 'linear', 'quadratic', 'exponential', 'logarithmic', 'sinusoidal'
    outliers: 0,
    show_outliers: true,
    show_equation: true,
    show_stats: true,
    show_grid: true,
    show_axes: true
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 800;
  const height = 500;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // Generate data with more options
  const data = DataSimulator.generateLinearData({
    n_samples: params.n_samples,
    noise: params.noise,
    slope: params.slope,
    intercept: params.intercept,
    trend: params.trend,
    outliers: params.outliers
  });
  
  // Find data bounds for scaling
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const xPadding = (xMax - xMin) * 0.1;
  const yPadding = (yMax - yMin) * 0.1;
  const bounds = {
    xMin: xMin - xPadding,
    xMax: xMax + xPadding,
    yMin: yMin - yPadding,
    yMax: yMax + yPadding
  };
  
  // Coordinate transformation functions
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 50);
  const toCanvasY = (y) => MathUtils.map(y, bounds.yMin, bounds.yMax, height - 50, 50);
  
  // Enhanced grid and axes drawing
  const drawGrid = () => {
    if (!params.show_grid && !params.show_axes) return;
    
    ctx.save();
    
    if (params.show_grid) {
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      
      // Vertical grid lines
      const xStep = Math.pow(10, Math.floor(Math.log10(bounds.xMax - bounds.xMin) / 2));
      for (let x = Math.ceil(bounds.xMin / xStep) * xStep; x <= Math.floor(bounds.xMax / xStep) * xStep; x += xStep) {
        const canvasX = toCanvasX(x);
        ctx.beginPath();
        ctx.moveTo(canvasX, toCanvasY(bounds.yMin));
        ctx.lineTo(canvasX, toCanvasY(bounds.yMax));
        ctx.stroke();
      }
      
      // Horizontal grid lines
      const yStep = Math.pow(10, Math.floor(Math.log10(bounds.yMax - bounds.yMin)) / 2);
      for (let y = Math.ceil(bounds.yMin / yStep) * yStep; y <= Math.floor(bounds.yMax / yStep) * yStep; y += yStep) {
        const canvasY = toCanvasY(y);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(bounds.xMin), canvasY);
        ctx.lineTo(toCanvasX(bounds.xMax), canvasY);
        ctx.stroke();
      }
      
      ctx.setLineDash([]);
    }
    
    if (params.show_axes) {
      // X axis
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(bounds.xMin), toCanvasY(0));
      ctx.lineTo(toCanvasX(bounds.xMax), toCanvasY(0));
      ctx.stroke();
      
      // Y axis
      ctx.beginPath();
      ctx.moveTo(toCanvasX(0), toCanvasY(bounds.yMin));
      ctx.lineTo(toCanvasX(0), toCanvasY(bounds.yMax));
      ctx.stroke();
      
      // Axis labels
      ctx.fillStyle = COLORS.text;
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('X', width - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Y', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing
  // In your drawing functions (e.g., drawDataPoints):
  const drawDataPoints = (points, ctx, toCanvasX, toCanvasY, progress = 1) => {
    ctx.save();
    
    // Batch draw circles - much faster than individual paths
    points.forEach(point => {
      const alpha = point.outlier ? 1 : progress; // Keep outliers fully visible
      const radius = point.outlier ? 7 : 5;
      
      ctx.beginPath();
      ctx.arc(
        toCanvasX(point.x), 
        toCanvasY(point.y), 
        radius * Math.min(progress, 1), // Scale radius during animation
        0, 
        Math.PI * 2
      );
      
      ctx.fillStyle = point.outlier 
        ? COLORS.accent 
        : COLORS.spectrum[point.label % COLORS.spectrum.length];
      ctx.globalAlpha = alpha;
      ctx.fill();
      
      if (point.outlier) {
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1.5;
        ctx.stroke();
      }
    });
    
    ctx.restore();
  };
  
  // Calculate linear regression coefficients with stats
  const calculateRegression = () => {
    const n = data.length;
    const sumX = data.reduce((sum, p) => sum + p.x, 0);
    const sumY = data.reduce((sum, p) => sum + p.y, 0);
    const sumXY = data.reduce((sum, p) => sum + p.x * p.y, 0);
    const sumXX = data.reduce((sum, p) => sum + p.x * p.x, 0);
    const sumYY = data.reduce((sum, p) => sum + p.y * p.y, 0);
    
    const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
    const intercept = (sumY - slope * sumX) / n;
    
    // Calculate R-squared
    const yMean = sumY / n;
    const ssTotal = data.reduce((sum, p) => sum + Math.pow(p.y - yMean, 2), 0);
    const ssResidual = data.reduce((sum, p) => {
      const yPred = intercept + slope * p.x;
      return sum + Math.pow(p.y - yPred, 2);
    }, 0);
    const rSquared = 1 - (ssResidual / ssTotal);
    
    // Calculate standard errors
    const mse = ssResidual / (n - 2);
    const slopeStdErr = Math.sqrt(mse / (sumXX - sumX * sumX / n));
    const interceptStdErr = Math.sqrt(mse * (1/n + sumX * sumX / (n * sumXX - sumX * sumX)));
    
    return { 
      slope, 
      intercept, 
      rSquared,
      slopeStdErr,
      interceptStdErr,
      ssTotal,
      ssResidual
    };
  };
  
  // Enhanced regression line drawing with confidence intervals
  const drawRegressionLine = (slope, intercept, progress = 1, stats = null) => {
    ctx.save();
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.globalAlpha = progress;
    
    const x1 = bounds.xMin;
    const y1 = intercept + slope * x1;
    const x2 = bounds.xMax;
    const y2 = intercept + slope * x2;
    
    ctx.beginPath();
    ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
    ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
    ctx.stroke();
    
    // Draw confidence interval if enabled
    if (params.show_confidence && stats) {
      const tValue = 1.96; // For 95% CI, assuming large n
      const n = data.length;
      const xMean = data.reduce((sum, p) => sum + p.x, 0) / n;
      const sxx = data.reduce((sum, p) => sum + Math.pow(p.x - xMean, 2), 0);
      
      // Calculate CI at each point along the line
      ctx.fillStyle = COLORS.accent + '30';
      ctx.strokeStyle = COLORS.accent + '60';
      ctx.lineWidth = 1;
      
      const steps = 20;
      const ciPoints = [];
      
      for (let i = 0; i <= steps; i++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / steps);
        const y = intercept + slope * x;
        
        // Standard error at this x
        const se = Math.sqrt(stats.ssResidual / (n - 2)) * 
                   Math.sqrt(1/n + Math.pow(x - xMean, 2) / sxx);
        
        ciPoints.push({
          x,
          yUpper: y + tValue * se,
          yLower: y - tValue * se
        });
      }
      
      // Draw confidence interval as a filled area
      ctx.beginPath();
      ctx.moveTo(toCanvasX(ciPoints[0].x), toCanvasY(ciPoints[0].yUpper));
      for (let i = 1; i < ciPoints.length; i++) {
        ctx.lineTo(toCanvasX(ciPoints[i].x), toCanvasY(ciPoints[i].yUpper));
      }
      for (let i = ciPoints.length - 1; i >= 0; i--) {
        ctx.lineTo(toCanvasX(ciPoints[i].x), toCanvasY(ciPoints[i].yLower));
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    
    // Draw prediction interval if enabled
    if (params.show_prediction && stats) {
      const tValue = 1.96; // For 95% PI, assuming large n
      const n = data.length;
      const xMean = data.reduce((sum, p) => sum + p.x, 0) / n;
      const sxx = data.reduce((sum, p) => sum + Math.pow(p.x - xMean, 2), 0);
      
      // Calculate PI at each point along the line
      ctx.fillStyle = COLORS.highlight + '20';
      ctx.strokeStyle = COLORS.highlight + '60';
      ctx.lineWidth = 1;
      
      const steps = 20;
      const piPoints = [];
      
      for (let i = 0; i <= steps; i++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / steps);
        const y = intercept + slope * x;
        
        // Standard error for prediction at this x
        const se = Math.sqrt(stats.ssResidual / (n - 2)) * 
                   Math.sqrt(1 + 1/n + Math.pow(x - xMean, 2) / sxx);
        
        piPoints.push({
          x,
          yUpper: y + tValue * se,
          yLower: y - tValue * se
        });
      }
      
      // Draw prediction interval as a filled area
      ctx.beginPath();
      ctx.moveTo(toCanvasX(piPoints[0].x), toCanvasY(piPoints[0].yUpper));
      for (let i = 1; i < piPoints.length; i++) {
        ctx.lineTo(toCanvasX(piPoints[i].x), toCanvasY(piPoints[i].yUpper));
      }
      for (let i = piPoints.length - 1; i >= 0; i--) {
        ctx.lineTo(toCanvasX(piPoints[i].x), toCanvasY(piPoints[i].yLower));
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }
    
    // Draw equation if enabled
    if (params.show_equation) {
      ctx.fillStyle = COLORS.text;
      ctx.font = '16px Arial';
      ctx.textAlign = 'left';
      
      let equationText;
      if (params.trend === 'linear') {
        equationText = `y = ${intercept.toFixed(2)} + ${slope.toFixed(2)}x`;
      } else if (params.trend === 'quadratic') {
        equationText = `y = ${intercept.toFixed(2)} + ${slope.toFixed(2)}x²`;
      } else if (params.trend === 'exponential') {
        equationText = `y = ${intercept.toFixed(2)} + e^(${slope.toFixed(2)}x)`;
      } else if (params.trend === 'logarithmic') {
        equationText = `y = ${intercept.toFixed(2)} + ${slope.toFixed(2)}ln(x)`;
      } else if (params.trend === 'sinusoidal') {
        equationText = `y = ${intercept.toFixed(2)} + ${slope.toFixed(2)}sin(x)`;
      }
      
      ctx.fillText(equationText, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 20);
      
      // Draw statistics if enabled
      if (params.show_stats && stats) {
        ctx.fillText(`R² = ${stats.rSquared.toFixed(3)}`, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 45);
        ctx.fillText(`Slope SE = ${stats.slopeStdErr.toFixed(3)}`, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 70);
        ctx.fillText(`Intercept SE = ${stats.interceptStdErr.toFixed(3)}`, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 95);
      }
    }
    
    ctx.restore();
  };
  
  // Enhanced residuals drawing
  const drawResiduals = (slope, intercept) => {
    if (!params.show_residuals) return;
    
    ctx.save();
    
    data.forEach(point => {
      const predictedY = intercept + slope * point.x;
      const canvasX = toCanvasX(point.x);
      const canvasYActual = toCanvasY(point.y);
      const canvasYPred = toCanvasY(predictedY);
      
      // Draw residual line
      ctx.strokeStyle = point.outlier ? COLORS.accent : COLORS.negative;
      ctx.lineWidth = point.outlier ? 2 : 1;
      ctx.setLineDash(point.outlier ? [] : [2, 2]);
      ctx.beginPath();
      ctx.moveTo(canvasX, canvasYActual);
      ctx.lineTo(canvasX, canvasYPred);
      ctx.stroke();
      
      // Highlight outliers
      if (point.outlier && params.show_outliers) {
        ctx.fillStyle = COLORS.accent + '40';
        ctx.beginPath();
        ctx.arc(canvasX, canvasYActual, 10, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    
    ctx.restore();
  };
  
  // Animate the regression line fitting with enhanced effects
  const animateRegressionFit = () => {
    const stats = calculateRegression();
    const { slope, intercept } = stats;
    
    // Initial random line
    let currentSlope = MathUtils.random(-5, 5);
    let currentIntercept = MathUtils.random(-5, 5);
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing
    data.forEach((point, i) => {
      timeline.add({
        duration: 300,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points that have appeared
          ctx.save();
          for (let j = 0; j <= i; j++) {
            const p = data[j];
            const alpha = j < i ? 1 : progress;
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            const radius = p.outlier && params.show_outliers ? 7 : 5;
            ctx.arc(toCanvasX(p.x), toCanvasY(p.y), radius, 0, Math.PI * 2);
            
            if (p.outlier && params.show_outliers) {
              ctx.fillStyle = COLORS.accent;
              ctx.strokeStyle = COLORS.text;
              ctx.lineWidth = 1.5;
              ctx.stroke();
            } else {
              ctx.fillStyle = COLORS.primary;
            }
            
            ctx.fill();
          }
          ctx.restore();
        }
      }, { delay: i * 20 });
    });
    
    // Phase 2: Animate line fitting
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid and points
        drawGrid();
        drawDataPoints();
        
        // Interpolate to final line
        currentSlope = MathUtils.lerp(currentSlope, slope, progress);
        currentIntercept = MathUtils.lerp(currentIntercept, intercept, progress);
        
        // Draw line and residuals
        drawRegressionLine(currentSlope, currentIntercept, progress, progress > 0.5 ? stats : null);
        
        // Show residuals in last part of animation
        if (progress > 0.7) {
          drawResiduals(currentSlope, currentIntercept);
        }
      }
    });
    
    // Phase 3: Final reveal with all elements
    timeline.add({
      duration: 500,
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints();
        drawRegressionLine(slope, intercept, 1, stats);
        drawResiduals(slope, intercept);
        
        // Pulse effect on equation
        if (params.show_equation) {
          ctx.save();
          ctx.fillStyle = COLORS.text;
          ctx.font = `16px Arial`;
          ctx.textAlign = 'left';
          
          let equationText;
          if (params.trend === 'linear') {
            equationText = `y = ${intercept.toFixed(2)} + ${slope.toFixed(2)}x`;
          } else if (params.trend === 'quadratic') {
            equationText = `y = ${intercept.toFixed(2)} + ${slope.toFixed(2)}x²`;
          } else if (params.trend === 'exponential') {
            equationText = `y = ${intercept.toFixed(2)} + e^(${slope.toFixed(2)}x)`;
          } else if (params.trend === 'logarithmic') {
            equationText = `y = ${intercept.toFixed(2)} + ${slope.toFixed(2)}ln(x)`;
          } else if (params.trend === 'sinusoidal') {
            equationText = `y = ${intercept.toFixed(2)} + ${slope.toFixed(2)}sin(x)`;
          }
          
          ctx.globalAlpha = progress;
          ctx.fillText(equationText, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 20);
          
          if (params.show_stats) {
            ctx.fillText(`R² = ${stats.rSquared.toFixed(3)}`, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 45);
            ctx.fillText(`Slope SE = ${stats.slopeStdErr.toFixed(3)}`, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 70);
            ctx.fillText(`Intercept SE = ${stats.interceptStdErr.toFixed(3)}`, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 95);
          }
          ctx.restore();
        }
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case '2d-scatter':
      params.show_residuals = false;
      params.show_confidence = false;
      params.show_prediction = false;
      animateRegressionFit();
      break;
      
    case 'residual-plot':
      params.show_residuals = true;
      params.show_confidence = false;
      params.show_prediction = false;
      animateRegressionFit();
      break;
      
    case 'confidence-intervals':
      params.show_residuals = true;
      params.show_confidence = true;
      params.show_prediction = false;
      animateRegressionFit();
      break;
      
    case 'prediction-intervals':
      params.show_residuals = false;
      params.show_confidence = false;
      params.show_prediction = true;
      animateRegressionFit();
      break;
      
    case 'all-intervals':
      params.show_residuals = true;
      params.show_confidence = true;
      params.show_prediction = true;
      animateRegressionFit();
      break;
      
    default:
      animateRegressionFit();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Samples',
        type: 'range',
        min: 20,
        max: 500,
        step: 10,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizeLinearRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Noise Level',
        type: 'range',
        min: 0,
        max: 2,
        step: 0.1,
        value: params.noise,
        onChange: (value) => {
          params.noise = parseFloat(value);
          visualizeLinearRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Outliers',
        type: 'range',
        min: 0,
        max: 20,
        step: 1,
        value: params.outliers,
        onChange: (value) => {
          params.outliers = parseInt(value);
          visualizeLinearRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Trend Type',
        type: 'select',
        options: [
          { value: 'linear', label: 'Linear', selected: params.trend === 'linear' },
          { value: 'quadratic', label: 'Quadratic', selected: params.trend === 'quadratic' },
          { value: 'exponential', label: 'Exponential', selected: params.trend === 'exponential' },
          { value: 'logarithmic', label: 'Logarithmic', selected: params.trend === 'logarithmic' },
          { value: 'sinusoidal', label: 'Sinusoidal', selected: params.trend === 'sinusoidal' }
        ],
        onChange: (value) => {
          params.trend = value;
          visualizeLinearRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Residuals',
        type: 'checkbox',
        checked: params.show_residuals,
        onChange: (value) => {
          params.show_residuals = value;
          visualizeLinearRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Confidence',
        type: 'checkbox',
        checked: params.show_confidence,
        onChange: (value) => {
          params.show_confidence = value;
          visualizeLinearRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Prediction',
        type: 'checkbox',
        checked: params.show_prediction,
        onChange: (value) => {
          params.show_prediction = value;
          visualizeLinearRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Stats',
        type: 'checkbox',
        checked: params.show_stats,
        onChange: (value) => {
          params.show_stats = value;
          visualizeLinearRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'default', label: 'Standard View', selected: visualizationType === 'default' },
          { value: '2d-scatter', label: '2D Scatter Plot', selected: visualizationType === '2d-scatter' },
          { value: 'residual-plot', label: 'Residual Plot', selected: visualizationType === 'residual-plot' },
          { value: 'confidence-intervals', label: 'Confidence Intervals', selected: visualizationType === 'confidence-intervals' },
          { value: 'prediction-intervals', label: 'Prediction Intervals', selected: visualizationType === 'prediction-intervals' },
          { value: 'all-intervals', label: 'All Intervals', selected: visualizationType === 'all-intervals' }
        ],
        onChange: (value) => {
          visualizeLinearRegression(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'Linear Regression Parameters',
      description: 'Adjust parameters to see how they affect the regression model.'
    });
  }
}

// =============================================
// Enhanced Logistic Regression Visualizations
// =============================================

function visualizeLogisticRegression(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 150,
    noise: 0.5,
    class_separation: 1.5,
    learning_rate: 0.1,
    iterations: 100,
    regularization: 0.01,
    show_boundary: true,
    show_probability: false,
    show_decision: false,
    show_loss: false,
    animation_duration: 2000,
    interactive: true,
    controlsContainer: null,
    distribution: 'linear', // 'linear', 'circular', 'xor'
    n_classes: 2,
    show_grid: true,
    show_axes: true
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 800;
  const height = 600;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // Generate data with more options
  const data = DataSimulator.generateLogisticData({
    n_samples: params.n_samples,
    noise: params.noise,
    class_separation: params.class_separation,
    distribution: params.distribution,
    n_classes: params.n_classes
  });
  
  // Find data bounds for scaling
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const padding = 1;
  const bounds = {
    xMin: xMin - padding,
    xMax: xMax + padding,
    yMin: yMin - padding,
    yMax: yMax + padding
  };
  
  // Coordinate transformation functions
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 50);
  const toCanvasY = (y) => MathUtils.map(y, bounds.yMin, bounds.yMax, height - 50, 50);
  
  // Enhanced grid and axes drawing
  const drawGrid = () => {
    if (!params.show_grid && !params.show_axes) return;
    
    ctx.save();
    
    if (params.show_grid) {
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      
      // Vertical grid lines
      for (let x = Math.ceil(bounds.xMin); x <= Math.floor(bounds.xMax); x++) {
        const canvasX = toCanvasX(x);
        ctx.beginPath();
        ctx.moveTo(canvasX, toCanvasY(bounds.yMin));
        ctx.lineTo(canvasX, toCanvasY(bounds.yMax));
        ctx.stroke();
      }
      
      // Horizontal grid lines
      for (let y = Math.ceil(bounds.yMin); y <= Math.floor(bounds.yMax); y++) {
        const canvasY = toCanvasY(y);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(bounds.xMin), canvasY);
        ctx.lineTo(toCanvasX(bounds.xMax), canvasY);
        ctx.stroke();
      }
      
      ctx.setLineDash([]);
    }
    
    if (params.show_axes) {
      // X axis
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(bounds.xMin), toCanvasY(0));
      ctx.lineTo(toCanvasX(bounds.xMax), toCanvasY(0));
      ctx.stroke();
      
      // Y axis
      ctx.beginPath();
      ctx.moveTo(toCanvasX(0), toCanvasY(bounds.yMin));
      ctx.lineTo(toCanvasX(0), toCanvasY(bounds.yMax));
      ctx.stroke();
      
      // Axis labels
      ctx.fillStyle = COLORS.text;
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Feature 1 (X)', width - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing
  const drawDataPoints = () => {
    ctx.save();
    data.forEach(point => {
      ctx.beginPath();
      const radius = 6;
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), radius, 0, Math.PI * 2);
      
      // Different colors for different classes
      const colors = COLORS.spectrum;
      ctx.fillStyle = colors[point.label % colors.length];
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    ctx.restore();
  };
  
  // Enhanced logistic regression model
  class LogisticModel {
    constructor() {
      this.weights = [MathUtils.random(-1, 1), MathUtils.random(-1, 1)]; // w1, w2
      this.bias = MathUtils.random(-1, 1); // b
      this.learningRate = params.learning_rate;
      this.regularization = params.regularization;
      this.lossHistory = [];
    }
    
    predict(x, y) {
      const z = this.weights[0] * x + this.weights[1] * y + this.bias;
      return MathUtils.sigmoid(z);
    }
    
    predictClass(x, y) {
      return this.predict(x, y) > 0.5 ? 1 : 0;
    }
    
    train(data, iterations) {
      const losses = [];
      const n = data.length;
      
      for (let i = 0; i < iterations; i++) {
        let dw1 = 0, dw2 = 0, db = 0;
        let loss = 0;
        
        data.forEach(point => {
          const { x, y, label } = point;
          const prediction = this.predict(x, y);
          
          // Cross-entropy loss with L2 regularization
          loss += -label * Math.log(prediction) - (1 - label) * Math.log(1 - prediction);
          
          // Gradients
          const error = prediction - label;
          dw1 += error * x;
          dw2 += error * y;
          db += error;
        });
        
        // Average gradients and add regularization
        dw1 = dw1 / n + this.regularization * this.weights[0];
        dw2 = dw2 / n + this.regularization * this.weights[1];
        db = db / n;
        loss = loss / n + this.regularization * (this.weights[0] ** 2 + this.weights[1] ** 2) / 2;
        
        // Update parameters
        this.weights[0] -= this.learningRate * dw1;
        this.weights[1] -= this.learningRate * dw2;
        this.bias -= this.learningRate * db;
        
        losses.push(loss);
      }
      
      this.lossHistory = [...this.lossHistory, ...losses];
      return losses;
    }
    
    getDecisionBoundary() {
      // Returns points for the decision boundary (where probability = 0.5)
      // Solve w1*x + w2*y + b = 0 => y = (-w1*x - b)/w2
      const x1 = bounds.xMin;
      const y1 = (-this.weights[0] * x1 - this.bias) / this.weights[1];
      const x2 = bounds.xMax;
      const y2 = (-this.weights[0] * x2 - this.bias) / this.weights[1];
      
      return { x1, y1, x2, y2 };
    }
  }
  
  // Draw decision boundary with enhanced styling
  const drawDecisionBoundary = (model, progress = 1) => {
    if (!params.show_boundary) return;
    
    ctx.save();
    ctx.strokeStyle = COLORS.primary;
    ctx.lineWidth = 3 * progress;
    ctx.globalAlpha = progress;
    ctx.setLineDash([5, 3]);
    
    const { x1, y1, x2, y2 } = model.getDecisionBoundary();
    
    ctx.beginPath();
    ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
    ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
    ctx.stroke();
    
    ctx.restore();
  };
  
  // Draw probability heatmap with enhanced styling
  const drawProbabilityHeatmap = (model) => {
    if (!params.show_probability) return;
    
    ctx.save();
    
    // Create gradient for probability colors
    const gradient = ctx.createLinearGradient(0, 0, width, 0);
    gradient.addColorStop(0, COLORS.accent);
    gradient.addColorStop(0.5, '#ffffff');
    gradient.addColorStop(1, COLORS.positive);
    
    // Draw probability for each pixel
    const resolution = 4; // pixels per grid cell
    const cellSize = resolution;
    
    for (let px = 0; px < width; px += cellSize) {
      for (let py = 0; py < height; py += cellSize) {
        // Convert canvas coordinates back to data coordinates
        const x = MathUtils.map(px, 50, width - 50, bounds.xMin, bounds.xMax);
        const y = MathUtils.map(py, height - 50, 50, bounds.yMin, bounds.yMax);
        
        const probability = model.predict(x, y);
        const alpha = Math.abs(probability - 0.5) * 2 * 0.3; // More transparent near boundary
        
        ctx.fillStyle = probability > 0.5 ? 
          `rgba(94, 160, 79, ${alpha})` : // positive class
          `rgba(237, 201, 72, ${alpha})`; // negative class
        
        ctx.fillRect(px, py, cellSize, cellSize);
      }
    }
    
    ctx.restore();
  };
  
  // Draw decision regions
  const drawDecisionRegions = (model) => {
    if (!params.show_decision) return;
    
    ctx.save();
    
    // Draw decision for each pixel
    const resolution = 8; // pixels per grid cell
    const cellSize = resolution;
    
    for (let px = 0; px < width; px += cellSize) {
      for (let py = 0; py < height; py += cellSize) {
        // Convert canvas coordinates back to data coordinates
        const x = MathUtils.map(px, 50, width - 50, bounds.xMin, bounds.xMax);
        const y = MathUtils.map(py, height - 50, 50, bounds.yMin, bounds.yMax);
        
        const decision = model.predictClass(x, y);
        const alpha = 0.1;
        
        ctx.fillStyle = decision === 1 ? 
          `rgba(94, 160, 79, ${alpha})` : // positive class
          `rgba(237, 201, 72, ${alpha})`; // negative class
        
        ctx.fillRect(px, py, cellSize, cellSize);
      }
    }
    
    ctx.restore();
  };
  
  // Draw loss curve
  const drawLossCurve = (model) => {
    if (!params.show_loss || model.lossHistory.length === 0) return;
    
    ctx.save();
    
    // Set up area for loss curve
    const lossWidth = 300;
    const lossHeight = 150;
    const lossX = width - lossWidth - 20;
    const lossY = 20;
    
    // Draw background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.fillRect(lossX, lossY, lossWidth, lossHeight);
    ctx.strokeRect(lossX, lossY, lossWidth, lossHeight);
    
    // Find min/max loss for scaling
    const maxLoss = Math.max(...model.lossHistory);
    const minLoss = Math.min(...model.lossHistory);
    const padding = (maxLoss - minLoss) * 0.1;
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(lossX + 10, lossY + lossHeight - 10);
    ctx.lineTo(lossX + 10, lossY + 10);
    ctx.lineTo(lossX + lossWidth - 10, lossY + 10);
    ctx.stroke();
    
    // Draw axis labels
    ctx.fillStyle = COLORS.text;
    ctx.font = '10px Arial';
    ctx.textAlign = 'right';
    ctx.fillText(maxLoss.toFixed(2), lossX + 8, lossY + 15);
    ctx.fillText(minLoss.toFixed(2), lossX + 8, lossY + lossHeight - 5);
    ctx.textAlign = 'center';
    ctx.fillText('0', lossX + 15, lossY + lossHeight - 2);
    ctx.fillText(model.lossHistory.length.toString(), lossX + lossWidth - 10, lossY + lossHeight - 2);
    ctx.fillText('Loss', lossX + 15, lossY + 8);
    ctx.fillText('Iteration', lossX + lossWidth - 10, lossY + lossHeight + 12);
    
    // Draw loss curve
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    model.lossHistory.forEach((loss, i) => {
      const x = lossX + 10 + (i / (model.lossHistory.length - 1)) * (lossWidth - 20);
      const y = lossY + 10 + ((loss - minLoss) / (maxLoss - minLoss)) * (lossHeight - 20);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw current loss marker
    const currentLoss = model.lossHistory[model.lossHistory.length - 1];
    const currentX = lossX + lossWidth - 10;
    const currentY = lossY + 10 + ((currentLoss - minLoss) / (maxLoss - minLoss)) * (lossHeight - 20);
    
    ctx.fillStyle = COLORS.accent;
    ctx.beginPath();
    ctx.arc(currentX, currentY, 4, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Loss: ${currentLoss.toFixed(4)}`, currentX + 8, currentY + 4);
    
    ctx.restore();
  };
  
  // Animate the logistic regression training with enhanced effects
  const animateLogisticFit = () => {
    const model = new LogisticModel();
    const steps = 20; // Number of animation steps
    const stepIterations = Math.ceil(params.iterations / steps);
    
    let currentStep = 0;
    let losses = [];
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing
    data.forEach((point, i) => {
      timeline.add({
        duration: 300,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          
          // Draw probability background if enabled
          if (params.show_probability) {
            drawProbabilityHeatmap(model);
          }
          
          // Draw decision regions if enabled
          if (params.show_decision) {
            drawDecisionRegions(model);
          }
          
          drawGrid();
          
          // Draw points that have appeared
          ctx.save();
          for (let j = 0; j <= i; j++) {
            const p = data[j];
            const alpha = j < i ? 1 : progress;
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 6, 0, Math.PI * 2);
            
            // Different colors for different classes
            const colors = COLORS.spectrum;
            ctx.fillStyle = colors[p.label % colors.length];
            ctx.fill();
            
            // Add outline for better visibility
            ctx.strokeStyle = COLORS.text;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
          ctx.restore();
        }
      }, { delay: i * 20 });
    });
    
    // Phase 2: Animate training process
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        const stepsCompleted = Math.floor(progress * steps);
        
        // Train model up to current step
        while (currentStep < stepsCompleted) {
          losses = [...losses, ...model.train(data, stepIterations)];
          currentStep++;
        }
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw probability background if enabled
        if (params.show_probability) {
          drawProbabilityHeatmap(model);
        }
        
        // Draw decision regions if enabled
        if (params.show_decision) {
          drawDecisionRegions(model);
        }
        
        // Draw grid and points
        drawGrid();
        drawDataPoints();
        
        // Draw decision boundary
        const currentProgress = (progress * steps) % 1;
        drawDecisionBoundary(model, currentProgress);
        
        // Draw loss curve if enabled
        if (params.show_loss) {
          drawLossCurve(model);
        }
      }
    });
    
    // Phase 3: Final reveal with all elements
    timeline.add({
      duration: 500,
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw probability background if enabled
        if (params.show_probability) {
          drawProbabilityHeatmap(model);
        }
        
        // Draw decision regions if enabled
        if (params.show_decision) {
          drawDecisionRegions(model);
        }
        
        // Draw grid and points
        drawGrid();
        drawDataPoints();
        
        // Draw decision boundary
        drawDecisionBoundary(model, 1);
        
        // Draw loss curve if enabled
        if (params.show_loss) {
          drawLossCurve(model);
        }
        
        // Draw equation
        ctx.save();
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        
        const { weights, bias } = model;
        const equationText = `P(y=1) = σ(${bias.toFixed(2)} + ${weights[0].toFixed(2)}x₁ + ${weights[1].toFixed(2)}x₂)`;
        
        ctx.fillText(equationText, 20, 30);
        ctx.restore();
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'decision-boundary':
      params.show_boundary = true;
      params.show_probability = false;
      params.show_decision = false;
      animateLogisticFit();
      break;
      
    case 'probability-surface':
      params.show_boundary = false;
      params.show_probability = true;
      params.show_decision = false;
      animateLogisticFit();
      break;
      
    case 'decision-regions':
      params.show_boundary = false;
      params.show_probability = false;
      params.show_decision = true;
      animateLogisticFit();
      break;
      
    case 'all':
      params.show_boundary = true;
      params.show_probability = true;
      params.show_decision = true;
      animateLogisticFit();
      break;
      
    case 'with-loss':
      params.show_boundary = true;
      params.show_probability = false;
      params.show_decision = false;
      params.show_loss = true;
      animateLogisticFit();
      break;
      
    default:
      animateLogisticFit();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Samples',
        type: 'range',
        min: 20,
        max: 500,
        step: 10,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Class Separation',
        type: 'range',
        min: 0.5,
        max: 3,
        step: 0.1,
        value: params.class_separation,
        onChange: (value) => {
          params.class_separation = parseFloat(value);
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Noise Level',
        type: 'range',
        min: 0,
        max: 1,
        step: 0.05,
        value: params.noise,
        onChange: (value) => {
          params.noise = parseFloat(value);
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Learning Rate',
        type: 'range',
        min: 0.01,
        max: 0.5,
        step: 0.01,
        value: params.learning_rate,
        onChange: (value) => {
          params.learning_rate = parseFloat(value);
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Regularization',
        type: 'range',
        min: 0,
        max: 0.1,
        step: 0.001,
        value: params.regularization,
        onChange: (value) => {
          params.regularization = parseFloat(value);
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Data Distribution',
        type: 'select',
        options: [
          { value: 'linear', label: 'Linear', selected: params.distribution === 'linear' },
          { value: 'circular', label: 'Circular', selected: params.distribution === 'circular' },
          { value: 'xor', label: 'XOR Pattern', selected: params.distribution === 'xor' }
        ],
        onChange: (value) => {
          params.distribution = value;
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Probability',
        type: 'checkbox',
        checked: params.show_probability,
        onChange: (value) => {
          params.show_probability = value;
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision',
        type: 'checkbox',
        checked: params.show_decision,
        onChange: (value) => {
          params.show_decision = value;
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Loss',
        type: 'checkbox',
        checked: params.show_loss,
        onChange: (value) => {
          params.show_loss = value;
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'default', label: 'Standard View', selected: visualizationType === 'default' },
          { value: 'decision-boundary', label: 'Decision Boundary', selected: visualizationType === 'decision-boundary' },
          { value: 'probability-surface', label: 'Probability Surface', selected: visualizationType === 'probability-surface' },
          { value: 'decision-regions', label: 'Decision Regions', selected: visualizationType === 'decision-regions' },
          { value: 'all', label: 'All Features', selected: visualizationType === 'all' },
          { value: 'with-loss', label: 'With Loss Curve', selected: visualizationType === 'with-loss' }
        ],
        onChange: (value) => {
          visualizeLogisticRegression(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'Logistic Regression Parameters',
      description: 'Adjust parameters to see how they affect the logistic regression model.'
    });
  }
}

// =============================================
// Enhanced Decision Tree Visualizations
// =============================================

function visualizeDecisionTree(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 200,
    n_classes: 3,
    max_depth: 3,
    min_samples_split: 2,
    min_samples_leaf: 1,
    show_tree: true,
    show_splits: true,
    show_regions: true,
    show_data: true,
    show_impurity: false,
    animation_duration: 2000,
    interactive: true,
    controlsContainer: null,
    distribution: 'concentric', // 'concentric', 'clusters', 'checkerboard'
    show_grid: true,
    show_axes: true
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 800;
  const height = 600;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // Generate data with more options
  const data = DataSimulator.generateTreeData({
    n_samples: params.n_samples,
    n_classes: params.n_classes,
    distribution: params.distribution
  });
  
  // Find data bounds for scaling
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const padding = 1;
  const bounds = {
    xMin: xMin - padding,
    xMax: xMax + padding,
    yMin: yMin - padding,
    yMax: yMax + padding
  };
  
  // Coordinate transformation functions
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 50);
  const toCanvasY = (y) => MathUtils.map(y, bounds.yMin, bounds.yMax, height - 50, 50);
  
  // Enhanced grid and axes drawing
  const drawGrid = () => {
    if (!params.show_grid && !params.show_axes) return;
    
    ctx.save();
    
    if (params.show_grid) {
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      
      // Vertical grid lines
      for (let x = Math.ceil(bounds.xMin); x <= Math.floor(bounds.xMax); x++) {
        const canvasX = toCanvasX(x);
        ctx.beginPath();
        ctx.moveTo(canvasX, toCanvasY(bounds.yMin));
        ctx.lineTo(canvasX, toCanvasY(bounds.yMax));
        ctx.stroke();
      }
      
      // Horizontal grid lines
      for (let y = Math.ceil(bounds.yMin); y <= Math.floor(bounds.yMax); y++) {
        const canvasY = toCanvasY(y);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(bounds.xMin), canvasY);
        ctx.lineTo(toCanvasX(bounds.xMax), canvasY);
        ctx.stroke();
      }
      
      ctx.setLineDash([]);
    }
    
    if (params.show_axes) {
      // X axis
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(bounds.xMin), toCanvasY(0));
      ctx.lineTo(toCanvasX(bounds.xMax), toCanvasY(0));
      ctx.stroke();
      
      // Y axis
      ctx.beginPath();
      ctx.moveTo(toCanvasX(0), toCanvasY(bounds.yMin));
      ctx.lineTo(toCanvasX(0), toCanvasY(bounds.yMax));
      ctx.stroke();
      
      // Axis labels
      ctx.fillStyle = COLORS.text;
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Feature 1 (X)', width - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing
  const drawDataPoints = () => {
    if (!params.show_data) return;
    
    ctx.save();
    data.forEach(point => {
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 5, 0, Math.PI * 2);
      
      // Different colors for different classes
      const colors = COLORS.spectrum;
      ctx.fillStyle = colors[point.label % colors.length];
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    ctx.restore();
  };
  
  // Enhanced decision tree node class
  class TreeNode {
    constructor(depth = 0, region = null) {
      this.depth = depth;
      this.left = null;
      this.right = null;
      this.feature = null;
      this.threshold = null;
      this.value = null;
      this.isLeaf = false;
      this.region = region;
      this.impurity = null;
      this.samples = null;
      this.classDistribution = null;
    }
    
    // Calculate Gini impurity
    giniImpurity(labels) {
      const counts = {};
      labels.forEach(label => {
        counts[label] = (counts[label] || 0) + 1;
      });
      
      let impurity = 1;
      Object.values(counts).forEach(count => {
        impurity -= Math.pow(count / labels.length, 2);
      });
      
      return impurity;
    }
    
    // Calculate class distribution
    calculateClassDistribution(labels) {
      const counts = {};
      labels.forEach(label => {
        counts[label] = (counts[label] || 0) + 1;
      });
      
      const distribution = {};
      const total = labels.length;
      Object.keys(counts).forEach(key => {
        distribution[key] = counts[key] / total;
      });
      
      return distribution;
    }
    
    // Find best split for the data
    findBestSplit(data) {
      let bestGini = Infinity;
      let bestFeature = null;
      let bestThreshold = null;
      let bestLeft = [];
      let bestRight = [];
      
      // Try all features
      ['x', 'y'].forEach(feature => {
        // Try multiple possible thresholds
        const values = data.map(p => p[feature]).sort((a, b) => a - b);
        const uniqueValues = [...new Set(values)];
        
        // Try potential thresholds (simplified for visualization)
        const nThresholds = Math.min(20, uniqueValues.length - 1);
        for (let i = 0; i < nThresholds; i++) {
          const idx = Math.floor(i * (uniqueValues.length - 1) / nThresholds);
          const threshold = uniqueValues[idx];
          
          // Split data
          const left = data.filter(p => p[feature] <= threshold);
          const right = data.filter(p => p[feature] > threshold);
          
          // Skip if split doesn't meet minimum samples
          if (left.length < params.min_samples_split || right.length < params.min_samples_split) {
            continue;
          }
          
          if (left.length < params.min_samples_leaf || right.length < params.min_samples_leaf) {
            continue;
          }
          
          // Calculate weighted Gini impurity
          const leftGini = this.giniImpurity(left.map(p => p.label));
          const rightGini = this.giniImpurity(right.map(p => p.label));
          const totalGini = (left.length * leftGini + right.length * rightGini) / data.length;
          
          // Update best split if better
          if (totalGini < bestGini) {
            bestGini = totalGini;
            bestFeature = feature;
            bestThreshold = threshold;
            bestLeft = left;
            bestRight = right;
          }
        }
      });
      
      return { 
        feature: bestFeature, 
        threshold: bestThreshold, 
        left: bestLeft, 
        right: bestRight,
        gini: bestGini
      };
    }
    
    // Build the tree recursively
    buildTree(data) {
      this.samples = data.length;
      this.impurity = this.giniImpurity(data.map(p => p.label));
      this.classDistribution = this.calculateClassDistribution(data.map(p => p.label));
      
      // Base cases
      if (this.depth >= params.max_depth || 
          data.length < params.min_samples_split * 2 || 
          new Set(data.map(p => p.label)).size === 1) {
        this.isLeaf = true;
        
        // Determine majority class
        const counts = {};
        data.forEach(p => {
          counts[p.label] = (counts[p.label] || 0) + 1;
        });
        
        this.value = Object.keys(counts).reduce((a, b) => counts[a] > counts[b] ? a : b);
        return;
      }
      
      // Find best split
      const { feature, threshold, left, right, gini } = this.findBestSplit(data);
      
      if (!feature) {
        this.isLeaf = true;
        this.value = data[0].label; // Default if no good split
        return;
      }
      
      this.feature = feature;
      this.threshold = threshold;
      
      // Create child nodes with updated regions
      let leftRegion = this.region ? { ...this.region } : { 
        xMin: bounds.xMin, 
        xMax: bounds.xMax, 
        yMin: bounds.yMin, 
        yMax: bounds.yMax 
      };
      
      let rightRegion = this.region ? { ...this.region } : { 
        xMin: bounds.xMin, 
        xMax: bounds.xMax, 
        yMin: bounds.yMin, 
        yMax: bounds.yMax 
      };
      
      if (feature === 'x') {
        leftRegion.xMax = threshold;
        rightRegion.xMin = threshold;
      } else {
        leftRegion.yMax = threshold;
        rightRegion.yMin = threshold;
      }
      
      this.left = new TreeNode(this.depth + 1, leftRegion);
      this.right = new TreeNode(this.depth + 1, rightRegion);
      
      // Recursively build tree
      this.left.buildTree(left);
      this.right.buildTree(right);
    }
    
    // Get all splits for visualization
    getSplits() {
      if (this.isLeaf) return [];
      
      const splits = [{
        feature: this.feature,
        threshold: this.threshold,
        depth: this.depth,
        impurity: this.impurity,
        samples: this.samples
      }];
      
      return [...splits, ...this.left.getSplits(), ...this.right.getSplits()];
    }
    
    // Get all regions for visualization
    getRegions() {
      if (this.isLeaf) {
        return [{ 
          region: this.region, 
          value: this.value,
          impurity: this.impurity,
          samples: this.samples,
          classDistribution: this.classDistribution
        }];
      }
      
      return [...this.left.getRegions(), ...this.right.getRegions()];
    }
    
    // Get tree structure for diagram
    getTreeStructure(x = 0.5, y = 0.1, width = 0.8, depth = 0) {
      const node = {
        x,
        y,
        feature: this.feature,
        threshold: this.threshold,
        value: this.value,
        isLeaf: this.isLeaf,
        impurity: this.impurity,
        samples: this.samples,
        classDistribution: this.classDistribution,
        children: []
      };
      
      if (!this.isLeaf) {
        const childWidth = width / 2;
        node.children.push(
          this.left.getTreeStructure(x - width/4, y + 0.2, childWidth, depth + 1)
        );
        node.children.push(
          this.right.getTreeStructure(x + width/4, y + 0.2, childWidth, depth + 1)
        );
      }
      
      return node;
    }
  }
  
  // Enhanced decision splits drawing
  const drawSplits = (splits, progress = 1) => {
    if (!params.show_splits) return;
    
    ctx.save();
    
    splits.forEach(split => {
      const alpha = MathUtils.clamp(progress * 2 - (split.depth / params.max_depth), 0, 1);
      if (alpha <= 0) return;
      
      ctx.strokeStyle = COLORS.primary;
      ctx.lineWidth = 2 + (1 - split.depth / params.max_depth) * 2; // Thicker lines for higher levels
      ctx.globalAlpha = alpha;
      
      if (split.feature === 'x') {
        // Vertical split
        const x = split.threshold;
        ctx.beginPath();
        ctx.moveTo(toCanvasX(x), toCanvasY(bounds.yMin));
        ctx.lineTo(toCanvasX(x), toCanvasY(bounds.yMax));
        ctx.stroke();
        
        // Draw split info if showing impurity
        if (params.show_impurity) {
          ctx.fillStyle = COLORS.text;
          ctx.font = '12px Arial';
          ctx.textAlign = 'center';
          ctx.fillText(
            `Gini: ${split.impurity.toFixed(2)} | Samples: ${split.samples}`, 
            toCanvasX(x), 
            toCanvasY(bounds.yMax) + 20
          );
        }
      } else {
        // Horizontal split
        const y = split.threshold;
        ctx.beginPath();
        ctx.moveTo(toCanvasX(bounds.xMin), toCanvasY(y));
        ctx.lineTo(toCanvasX(bounds.xMax), toCanvasY(y));
        ctx.stroke();
        
        // Draw split info if showing impurity
        if (params.show_impurity) {
          ctx.fillStyle = COLORS.text;
          ctx.font = '12px Arial';
          ctx.textAlign = 'right';
          ctx.fillText(
            `Gini: ${split.impurity.toFixed(2)} | Samples: ${split.samples}`, 
            toCanvasX(bounds.xMin) - 10, 
            toCanvasY(y)
          );
        }
      }
    });
    
    ctx.restore();
  };
  
  // Enhanced decision regions drawing
  const drawRegions = (regions) => {
    if (!params.show_regions) return;
    
    ctx.save();
    
    const colors = COLORS.spectrum;
    
    regions.forEach(region => {
      const { xMin, xMax, yMin, yMax, value, impurity, samples, classDistribution } = region;
      
      // Fill region with class color
      ctx.fillStyle = colors[value % colors.length] + '40'; // Add transparency
      ctx.fillRect(
        toCanvasX(xMin),
        toCanvasY(yMax),
        toCanvasX(xMax) - toCanvasX(xMin),
        toCanvasY(yMin) - toCanvasY(yMax)
      );
      
      // Draw region info if showing impurity
      if (params.show_impurity) {
        ctx.fillStyle = COLORS.text;
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        
        const centerX = (xMin + xMax) / 2;
        const centerY = (yMin + yMax) / 2;
        
        ctx.fillText(
          `Class: ${value} | Gini: ${impurity.toFixed(2)}`, 
          toCanvasX(centerX), 
          toCanvasY(centerY)
        );
        
        ctx.fillText(
          `Samples: ${samples}`, 
          toCanvasX(centerX), 
          toCanvasY(centerY) + 12
        );
      }
    });
    
    ctx.restore();
  };
  
  // Enhanced tree diagram drawing
  const drawTreeDiagram = (tree) => {
    if (!params.show_tree) return;
    
    ctx.save();
    
    // Get tree structure
    const treeStructure = tree.getTreeStructure();
    
    // Draw tree recursively
    const drawNode = (node) => {
      const nodeX = width * node.x;
      const nodeY = height * node.y;
      const nodeRadius = 20;
      
      // Draw node
      ctx.fillStyle = '#ffffff';
      ctx.strokeStyle = COLORS.primary;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(nodeX, nodeY, nodeRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      
      // Draw node content
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      
      if (node.isLeaf) {
        // Leaf node - show class
        ctx.fillText(node.value.toString(), nodeX, nodeY + 4);
      } else {
        // Decision node - show feature and threshold
        ctx.fillText(`${node.feature} ≤ ${node.threshold.toFixed(2)}`, nodeX, nodeY + 4);
      }
      
      // Draw node info if showing impurity
      if (params.show_impurity) {
        ctx.font = '10px Arial';
        ctx.fillText(`Gini: ${node.impurity.toFixed(2)}`, nodeX, nodeY + nodeRadius + 12);
        ctx.fillText(`Samples: ${node.samples}`, nodeX, nodeY + nodeRadius + 24);
      }
      
      // Draw connections to children
      node.children.forEach(child => {
        const childX = width * child.x;
        const childY = height * child.y;
        
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(nodeX, nodeY + nodeRadius);
        ctx.lineTo(childX, childY - nodeRadius);
        ctx.stroke();
        
        // Draw child nodes
        drawNode(child);
      });
    };
    
    // Start drawing from root
    drawNode(treeStructure);
    
    // Draw legend
    const legendItems = [];
    for (let i = 0; i < params.n_classes; i++) {
      legendItems.push({
        label: `Class ${i}`,
        color: colors[i % colors.length]
      });
    }
    
    const legend = DomUtils.createLegend(legendItems, {
      position: 'absolute',
      top: '20px',
      left: '20px'
    });
    
    // Temporarily add legend to canvas
    const legendCanvas = document.createElement('canvas');
    legendCanvas.width = 200;
    legendCanvas.height = 100;
    const legendCtx = legendCanvas.getContext('2d');
    
    // Render legend to canvas
    legendCtx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    legendCtx.fillRect(0, 0, legendCanvas.width, legendCanvas.height);
    legendCtx.fillStyle = COLORS.text;
    legendCtx.font = '14px Arial';
    legendCtx.textAlign = 'left';
    legendCtx.fillText('Classes', 10, 20);
    
    legendItems.forEach((item, i) => {
      legendCtx.fillStyle = item.color;
      legendCtx.beginPath();
      legendCtx.arc(15, 35 + i * 20, 6, 0, Math.PI * 2);
      legendCtx.fill();
      
      legendCtx.fillStyle = COLORS.text;
      legendCtx.fillText(item.label, 30, 40 + i * 20);
    });
    
    // Draw legend onto main canvas
    ctx.drawImage(legendCanvas, 20, 20);
    
    ctx.restore();
  };
  
  // Animate the decision tree building with enhanced effects
  const animateTreeBuilding = () => {
    // Build the tree
    const tree = new TreeNode();
    tree.buildTree(data);
    
    // Get splits and regions for visualization
    const splits = tree.getSplits();
    const regions = tree.getRegions();
    
    // Animation steps
    const steps = params.max_depth + 1;
    let currentStep = 0;
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing
    data.forEach((point, i) => {
      timeline.add({
        duration: 300,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          
          // Draw regions if enabled
          if (params.show_regions) {
            drawRegions(regions);
          }
          
          drawGrid();
          
          // Draw points that have appeared
          ctx.save();
          for (let j = 0; j <= i; j++) {
            const p = data[j];
            const alpha = j < i ? 1 : progress;
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 5, 0, Math.PI * 2);
            
            // Different colors for different classes
            const colors = COLORS.spectrum;
            ctx.fillStyle = colors[p.label % colors.length];
            ctx.fill();
            
            // Add outline for better visibility
            ctx.strokeStyle = COLORS.text;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
          ctx.restore();
        }
      }, { delay: i * 10 });
    });
    
    // Phase 2: Animate tree building
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        const stepsCompleted = Math.floor(progress * steps);
        
        // Update current step
        currentStep = stepsCompleted;
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw regions if enabled
        if (params.show_regions) {
          drawRegions(regions);
        }
        
        // Draw grid and points
        drawGrid();
        drawDataPoints();
        
        // Draw splits up to current depth
        const visibleSplits = splits.filter(split => split.depth < currentStep);
        drawSplits(visibleSplits, progress);
        
        // Draw tree diagram if enabled
        if (params.show_tree) {
          drawTreeDiagram(tree);
        }
      }
    });
    
    // Phase 3: Final reveal with all elements
    timeline.add({
      duration: 500,
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw regions if enabled
        if (params.show_regions) {
          drawRegions(regions);
        }
        
        // Draw grid and points
        drawGrid();
        drawDataPoints();
        
        // Draw all splits
        drawSplits(splits, 1);
        
        // Draw tree diagram if enabled
        if (params.show_tree) {
          drawTreeDiagram(tree);
        }
        
        // Draw title
        ctx.save();
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Decision Tree (max_depth=${params.max_depth})`, 20, 30);
        ctx.restore();
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'tree-split':
      params.show_splits = true;
      params.show_regions = false;
      params.show_tree = false;
      animateTreeBuilding();
      break;
      
    case 'tree-diagram':
      params.show_tree = true;
      params.show_splits = false;
      params.show_regions = false;
      animateTreeBuilding();
      break;
      
    case 'decision-boundary':
      params.show_regions = true;
      params.show_splits = false;
      params.show_tree = false;
      animateTreeBuilding();
      break;
      
    case 'all':
      params.show_splits = true;
      params.show_regions = true;
      params.show_tree = true;
      animateTreeBuilding();
      break;
      
    case 'with-impurity':
      params.show_splits = true;
      params.show_regions = true;
      params.show_tree = true;
      params.show_impurity = true;
      animateTreeBuilding();
      break;
      
    default:
      animateTreeBuilding();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Max Depth',
        type: 'range',
        min: 1,
        max: 6,
        step: 1,
        value: params.max_depth,
        onChange: (value) => {
          params.max_depth = parseInt(value);
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Min Samples Split',
        type: 'range',
        min: 2,
        max: 20,
        step: 1,
        value: params.min_samples_split,
        onChange: (value) => {
          params.min_samples_split = parseInt(value);
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Min Samples Leaf',
        type: 'range',
        min: 1,
        max: 10,
        step: 1,
        value: params.min_samples_leaf,
        onChange: (value) => {
          params.min_samples_leaf = parseInt(value);
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Data Distribution',
        type: 'select',
        options: [
          { value: 'concentric', label: 'Concentric', selected: params.distribution === 'concentric' },
          { value: 'clusters', label: 'Clusters', selected: params.distribution === 'clusters' },
          { value: 'checkerboard', label: 'Checkerboard', selected: params.distribution === 'checkerboard' }
        ],
        onChange: (value) => {
          params.distribution = value;
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Splits',
        type: 'checkbox',
        checked: params.show_splits,
        onChange: (value) => {
          params.show_splits = value;
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Regions',
        type: 'checkbox',
        checked: params.show_regions,
        onChange: (value) => {
          params.show_regions = value;
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Tree',
        type: 'checkbox',
        checked: params.show_tree,
        onChange: (value) => {
          params.show_tree = value;
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Impurity',
        type: 'checkbox',
        checked: params.show_impurity,
        onChange: (value) => {
          params.show_impurity = value;
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'default', label: 'Standard View', selected: visualizationType === 'default' },
          { value: 'tree-split', label: 'Tree Splits', selected: visualizationType === 'tree-split' },
          { value: 'tree-diagram', label: 'Tree Diagram', selected: visualizationType === 'tree-diagram' },
          { value: 'decision-boundary', label: 'Decision Boundary', selected: visualizationType === 'decision-boundary' },
          { value: 'all', label: 'All Features', selected: visualizationType === 'all' },
          { value: 'with-impurity', label: 'With Impurity', selected: visualizationType === 'with-impurity' }
        ],
        onChange: (value) => {
          visualizeDecisionTree(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'Decision Tree Parameters',
      description: 'Adjust parameters to see how they affect the decision tree model.'
    });
  }
}

// =============================================
// K-Means Clustering Visualizations
// =============================================

function visualizeKMeans(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 200,
    n_clusters: 3,
    max_iterations: 10,
    show_centroids: true,
    show_voronoi: false,
    show_convergence: false,
    animation_duration: 2000,
    interactive: true,
    controlsContainer: null,
    distribution: 'blobs', // 'blobs', 'moons', 'circles'
    cluster_std: 0.5,
    show_grid: true,
    show_axes: true
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 800;
  const height = 600;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // Generate data with more options
  const data = DataSimulator.generateClusteringData({
    n_samples: params.n_samples,
    n_clusters: params.n_clusters,
    distribution: params.distribution,
    cluster_std: params.cluster_std
  });
  
  // Find data bounds for scaling
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const padding = 1;
  const bounds = {
    xMin: xMin - padding,
    xMax: xMax + padding,
    yMin: yMin - padding,
    yMax: yMax + padding
  };
  
  // Coordinate transformation functions
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 50);
  const toCanvasY = (y) => MathUtils.map(y, bounds.yMin, bounds.yMax, height - 50, 50);
  
  // Enhanced grid and axes drawing
  const drawGrid = () => {
    if (!params.show_grid && !params.show_axes) return;
    
    ctx.save();
    
    if (params.show_grid) {
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      
      // Vertical grid lines
      for (let x = Math.ceil(bounds.xMin); x <= Math.floor(bounds.xMax); x++) {
        const canvasX = toCanvasX(x);
        ctx.beginPath();
        ctx.moveTo(canvasX, toCanvasY(bounds.yMin));
        ctx.lineTo(canvasX, toCanvasY(bounds.yMax));
        ctx.stroke();
      }
      
      // Horizontal grid lines
      for (let y = Math.ceil(bounds.yMin); y <= Math.floor(bounds.yMax); y++) {
        const canvasY = toCanvasY(y);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(bounds.xMin), canvasY);
        ctx.lineTo(toCanvasX(bounds.xMax), canvasY);
        ctx.stroke();
      }
      
      ctx.setLineDash([]);
    }
    
    if (params.show_axes) {
      // X axis
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(bounds.xMin), toCanvasY(0));
      ctx.lineTo(toCanvasX(bounds.xMax), toCanvasY(0));
      ctx.stroke();
      
      // Y axis
      ctx.beginPath();
      ctx.moveTo(toCanvasX(0), toCanvasY(bounds.yMin));
      ctx.lineTo(toCanvasX(0), toCanvasY(bounds.yMax));
      ctx.stroke();
      
      // Axis labels
      ctx.fillStyle = COLORS.text;
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Feature 1 (X)', width - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing
  const drawDataPoints = (assignments = null) => {
    ctx.save();
    
    data.forEach((point, i) => {
      const clusterIdx = assignments ? assignments[i] : point.trueCluster;
      
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 5, 0, Math.PI * 2);
      
      // Different colors for different clusters
      const colors = COLORS.spectrum;
      ctx.fillStyle = colors[clusterIdx % colors.length];
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    
    ctx.restore();
  };
  
  // K-Means clustering algorithm
  class KMeans {
    constructor(k) {
      this.k = k;
      this.centroids = [];
      this.history = [];
      this.inertiaHistory = [];
    }
    
    // Initialize centroids randomly
    initializeCentroids(data) {
      // Select k random points as initial centroids
      const shuffled = [...data].sort(() => 0.5 - Math.random());
      this.centroids = shuffled.slice(0, this.k).map(p => ({
        x: p.x,
        y: p.y,
        cluster: null
      }));
      
      // Assign cluster indices
      this.centroids.forEach((c, i) => c.cluster = i);
      
      this.history = [[...this.centroids]];
      this.inertiaHistory = [];
    }
    
    // Assign each point to the nearest centroid
    assignPoints(data) {
      const assignments = [];
      let inertia = 0;
      
      data.forEach(point => {
        let minDist = Infinity;
        let cluster = 0;
        
        // Find nearest centroid
        this.centroids.forEach((centroid, i) => {
          const dist = MathUtils.distance(point.x, point.y, centroid.x, centroid.y);
          if (dist < minDist) {
            minDist = dist;
            cluster = i;
          }
        });
        
        assignments.push(cluster);
        inertia += minDist * minDist;
      });
      
      this.inertiaHistory.push(inertia);
      return assignments;
    }
    
    // Update centroids to be the mean of their assigned points
    updateCentroids(data, assignments) {
      const newCentroids = [];
      const clusterSums = Array(this.k).fill().map(() => ({ x: 0, y: 0, count: 0 }));
      
      // Sum points for each cluster
      data.forEach((point, i) => {
        const cluster = assignments[i];
        clusterSums[cluster].x += point.x;
        clusterSums[cluster].y += point.y;
        clusterSums[cluster].count++;
      });
      
      // Calculate new centroids
      clusterSums.forEach((sum, i) => {
        if (sum.count > 0) {
          newCentroids.push({
            x: sum.x / sum.count,
            y: sum.y / sum.count,
            cluster: i
          });
        } else {
          // If a cluster has no points, keep the old centroid
          newCentroids.push({ ...this.centroids[i] });
        }
      });
      
      this.centroids = newCentroids;
      this.history.push([...this.centroids]);
    }
    
    // Run the algorithm for a number of iterations
    fit(data, iterations) {
      this.initializeCentroids(data);
      
      for (let i = 0; i < iterations; i++) {
        const assignments = this.assignPoints(data);
        this.updateCentroids(data, assignments);
      }
      
      return this.assignPoints(data);
    }
    
    // Calculate Voronoi diagram edges
    getVoronoiEdges() {
      if (this.centroids.length < 2) return [];
      
      const edges = [];
      
      // For each pair of centroids, find the perpendicular bisector
      for (let i = 0; i < this.centroids.length; i++) {
        for (let j = i + 1; j < this.centroids.length; j++) {
          const c1 = this.centroids[i];
          const c2 = this.centroids[j];
          
          // Midpoint
          const mx = (c1.x + c2.x) / 2;
          const my = (c1.y + c2.y) / 2;
          
          // Slope of line between centroids
          const dx = c2.x - c1.x;
          const dy = c2.y - c1.y;
          
          // Skip if points are coincident
          if (dx === 0 && dy === 0) continue;
          
          // Slope of perpendicular bisector
          let px, py;
          if (dx === 0) {
            // Vertical line, horizontal bisector
            px = 1; py = 0;
          } else if (dy === 0) {
            // Horizontal line, vertical bisector
            px = 0; py = 1;
          } else {
            // General case
            const slope = dy / dx;
            const perpSlope = -1 / slope;
            px = 1;
            py = perpSlope;
          }
          
          // Normalize direction vector
          const length = Math.sqrt(px * px + py * py);
          px /= length;
          py /= length;
          
          // Extend line to bounds
          const tValues = [];
          
          // Intersection with left boundary (x = bounds.xMin)
          let t = (bounds.xMin - mx) / px;
          let y = my + t * py;
          if (y >= bounds.yMin && y <= bounds.yMax) tValues.push(t);
          
          // Intersection with right boundary (x = bounds.xMax)
          t = (bounds.xMax - mx) / px;
          y = my + t * py;
          if (y >= bounds.yMin && y <= bounds.yMax) tValues.push(t);
          
          // Intersection with bottom boundary (y = bounds.yMin)
          t = (bounds.yMin - my) / py;
          let x = mx + t * px;
          if (x >= bounds.xMin && x <= bounds.xMax) tValues.push(t);
          
          // Intersection with top boundary (y = bounds.yMax)
          t = (bounds.yMax - my) / py;
          x = mx + t * px;
          if (x >= bounds.xMin && x <= bounds.xMax) tValues.push(t);
          
          // If we have two intersections, draw the line segment
          if (tValues.length >= 2) {
            tValues.sort((a, b) => a - b);
            const t1 = tValues[0];
            const t2 = tValues[tValues.length - 1];
            
            edges.push({
              x1: mx + t1 * px,
              y1: my + t1 * py,
              x2: mx + t2 * px,
              y2: my + t2 * py
            });
          }
        }
      }
      
      return edges;
    }
  }
  
  // Draw centroids with enhanced styling
  const drawCentroids = (centroids, progress = 1) => {
    if (!params.show_centroids) return;
    
    ctx.save();
    
    centroids.forEach(centroid => {
      const alpha = Math.min(progress * 2, 1);
      
      // Draw centroid
      ctx.fillStyle = COLORS.spectrum[centroid.cluster % COLORS.spectrum.length];
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(toCanvasX(centroid.x), toCanvasY(centroid.y), 10, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw outline
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw cluster number
      ctx.fillStyle = COLORS.text;
      ctx.font = 'bold 12px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(centroid.cluster.toString(), toCanvasX(centroid.x), toCanvasY(centroid.y));
      
      // Draw movement path if history exists
      if (progress < 1 && this.kmeans && this.kmeans.history.length > 1) {
        const history = this.kmeans.history;
        const currentIter = Math.floor(progress * (history.length - 1));
        const nextIter = Math.min(currentIter + 1, history.length - 1);
        const interp = (progress * (history.length - 1)) % 1;
        
        const prevCentroid = history[currentIter].find(c => c.cluster === centroid.cluster);
        const nextCentroid = history[nextIter].find(c => c.cluster === centroid.cluster);
        
        if (prevCentroid && nextCentroid) {
          const x = MathUtils.lerp(prevCentroid.x, nextCentroid.x, interp);
          const y = MathUtils.lerp(prevCentroid.y, nextCentroid.y, interp);
          
          // Draw path
          ctx.strokeStyle = COLORS.spectrum[centroid.cluster % COLORS.spectrum.length] + '80';
          ctx.lineWidth = 2;
          ctx.setLineDash([3, 3]);
          ctx.beginPath();
          ctx.moveTo(toCanvasX(prevCentroid.x), toCanvasY(prevCentroid.y));
          ctx.lineTo(toCanvasX(x), toCanvasY(y));
          ctx.stroke();
          ctx.setLineDash([]);
        }
      }
    });
    
    ctx.restore();
  };
  
  // Draw Voronoi diagram
  const drawVoronoi = (edges) => {
    if (!params.show_voronoi) return;
    
    ctx.save();
    
    ctx.strokeStyle = COLORS.primary + '80';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    
    edges.forEach(edge => {
      ctx.beginPath();
      ctx.moveTo(toCanvasX(edge.x1), toCanvasY(edge.y1));
      ctx.lineTo(toCanvasX(edge.x2), toCanvasY(edge.y2));
      ctx.stroke();
    });
    
    ctx.setLineDash([]);
    ctx.restore();
  };
  
  // Draw inertia/convergence plot
  const drawConvergence = (inertiaHistory) => {
    if (!params.show_convergence || inertiaHistory.length === 0) return;
    
    ctx.save();
    
    // Set up area for convergence plot
    const plotWidth = 300;
    const plotHeight = 150;
    const plotX = width - plotWidth - 20;
    const plotY = 20;
    
    // Draw background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.fillRect(plotX, plotY, plotWidth, plotHeight);
    ctx.strokeRect(plotX, plotY, plotWidth, plotHeight);
    
    // Find min/max inertia for scaling
    const maxInertia = Math.max(...inertiaHistory);
    const minInertia = Math.min(...inertiaHistory);
    const padding = (maxInertia - minInertia) * 0.1;
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(plotX + 10, plotY + plotHeight - 10);
    ctx.lineTo(plotX + 10, plotY + 10);
    ctx.lineTo(plotX + plotWidth - 10, plotY + 10);
    ctx.stroke();
    
    // Draw axis labels
    ctx.fillStyle = COLORS.text;
    ctx.font = '10px Arial';
    ctx.textAlign = 'right';
    ctx.fillText(maxInertia.toFixed(0), plotX + 8, plotY + 15);
    ctx.fillText(minInertia.toFixed(0), plotX + 8, plotY + plotHeight - 5);
    ctx.textAlign = 'center';
    ctx.fillText('0', plotX + 15, plotY + plotHeight - 2);
    ctx.fillText(inertiaHistory.length.toString(), plotX + plotWidth - 10, plotY + plotHeight - 2);
    ctx.fillText('Inertia', plotX + 15, plotY + 8);
    ctx.fillText('Iteration', plotX + plotWidth - 10, plotY + plotHeight + 12);
    
    // Draw convergence curve
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    inertiaHistory.forEach((inertia, i) => {
      const x = plotX + 10 + (i / (inertiaHistory.length - 1)) * (plotWidth - 20);
      const y = plotY + 10 + ((inertia - minInertia) / (maxInertia - minInertia)) * (plotHeight - 20);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw current inertia marker
    const currentInertia = inertiaHistory[inertiaHistory.length - 1];
    const currentX = plotX + plotWidth - 10;
    const currentY = plotY + 10 + ((currentInertia - minInertia) / (maxInertia - minInertia)) * (plotHeight - 20);
    
    ctx.fillStyle = COLORS.accent;
    ctx.beginPath();
    ctx.arc(currentX, currentY, 4, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Inertia: ${currentInertia.toFixed(2)}`, currentX + 8, currentY + 4);
    
    ctx.restore();
  };
  
  // Animate the K-Means clustering with enhanced effects
  const animateKMeans = () => {
    this.kmeans = new KMeans(params.n_clusters);
    const steps = params.max_iterations;
    let currentStep = 0;
    let assignments = [];
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing
    data.forEach((point, i) => {
      timeline.add({
        duration: 300,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points that have appeared
          ctx.save();
          for (let j = 0; j <= i; j++) {
            const p = data[j];
            const alpha = j < i ? 1 : progress;
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 5, 0, Math.PI * 2);
            
            // Different colors for different clusters
            const colors = COLORS.spectrum;
            ctx.fillStyle = colors[p.trueCluster % colors.length];
            ctx.fill();
            
            // Add outline for better visibility
            ctx.strokeStyle = COLORS.text;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
          ctx.restore();
        }
      }, { delay: i * 10 });
    });
    
    // Phase 2: Animate K-Means iterations
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        const stepsCompleted = Math.floor(progress * steps);
        
        // Run iterations up to current step
        while (currentStep < stepsCompleted) {
          assignments = this.kmeans.fit(data, 1);
          currentStep++;
        }
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid and points
        drawGrid();
        drawDataPoints(assignments);
        
        // Draw centroids
        const interp = (progress * steps) % 1;
        drawCentroids(this.kmeans.centroids, interp);
        
        // Draw Voronoi diagram if enabled
        if (params.show_voronoi) {
          const edges = this.kmeans.getVoronoiEdges();
          drawVoronoi(edges);
        }
        
        // Draw convergence plot if enabled
        if (params.show_convergence) {
          drawConvergence(this.kmeans.inertiaHistory);
        }
      }
    });
    
    // Phase 3: Final reveal with all elements
    timeline.add({
      duration: 500,
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid and points
        drawGrid();
        drawDataPoints(assignments);
        
        // Draw centroids
        drawCentroids(this.kmeans.centroids, 1);
        
        // Draw Voronoi diagram if enabled
        if (params.show_voronoi) {
          const edges = this.kmeans.getVoronoiEdges();
          drawVoronoi(edges);
        }
        
        // Draw convergence plot if enabled
        if (params.show_convergence) {
          drawConvergence(this.kmeans.inertiaHistory);
        }
        
        // Draw title
        ctx.save();
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`K-Means Clustering (k=${params.n_clusters})`, 20, 30);
        ctx.restore();
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'centroids-only':
      params.show_centroids = true;
      params.show_voronoi = false;
      params.show_convergence = false;
      animateKMeans();
      break;
      
    case 'with-voronoi':
      params.show_centroids = true;
      params.show_voronoi = true;
      params.show_convergence = false;
      animateKMeans();
      break;
      
    case 'with-convergence':
      params.show_centroids = true;
      params.show_voronoi = false;
      params.show_convergence = true;
      animateKMeans();
      break;
      
    case 'all':
      params.show_centroids = true;
      params.show_voronoi = true;
      params.show_convergence = true;
      animateKMeans();
      break;
      
    default:
      animateKMeans();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Clusters (k)',
        type: 'range',
        min: 2,
        max: 8,
        step: 1,
        value: params.n_clusters,
        onChange: (value) => {
          params.n_clusters = parseInt(value);
          visualizeKMeans(containerId, visualizationType, params);
        }
      },
      {
        label: 'Max Iterations',
        type: 'range',
        min: 1,
        max: 20,
        step: 1,
        value: params.max_iterations,
        onChange: (value) => {
          params.max_iterations = parseInt(value);
          visualizeKMeans(containerId, visualizationType, params);
        }
      },
      {
        label: 'Data Distribution',
        type: 'select',
        options: [
          { value: 'blobs', label: 'Gaussian Blobs', selected: params.distribution === 'blobs' },
          { value: 'moons', label: 'Moons', selected: params.distribution === 'moons' },
          { value: 'circles', label: 'Circles', selected: params.distribution === 'circles' }
        ],
        onChange: (value) => {
          params.distribution = value;
          visualizeKMeans(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Centroids',
        type: 'checkbox',
        checked: params.show_centroids,
        onChange: (value) => {
          params.show_centroids = value;
          visualizeKMeans(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Voronoi',
        type: 'checkbox',
        checked: params.show_voronoi,
        onChange: (value) => {
          params.show_voronoi = value;
          visualizeKMeans(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Convergence',
        type: 'checkbox',
        checked: params.show_convergence,
        onChange: (value) => {
          params.show_convergence = value;
          visualizeKMeans(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'default', label: 'Standard View', selected: visualizationType === 'default' },
          { value: 'centroids-only', label: 'Centroids Only', selected: visualizationType === 'centroids-only' },
          { value: 'with-voronoi', label: 'With Voronoi', selected: visualizationType === 'with-voronoi' },
          { value: 'with-convergence', label: 'With Convergence', selected: visualizationType === 'with-convergence' },
          { value: 'all', label: 'All Features', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizeKMeans(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'K-Means Parameters',
      description: 'Adjust parameters to see how they affect the K-Means clustering.'
    });
  }
}

// =============================================
// Neural Network Visualizations
// =============================================

function visualizeNeuralNetwork(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 200,
    n_classes: 3,
    learning_rate: 0.1,
    iterations: 100,
    regularization: 0.01,
    hidden_layers: 1,
    hidden_units: 5,
    activation: 'relu',
    show_decision: true,
    show_loss: false,
    show_network: false,
    animation_duration: 2000,
    interactive: true,
    controlsContainer: null,
    show_grid: true,
    show_axes: true
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 800;
  const height = 600;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // Generate data with more options
  const data = DataSimulator.generateNeuralNetworkData({
    n_samples: params.n_samples,
    n_classes: params.n_classes,
    complexity: 2
  });
  
  // Find data bounds for scaling
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const padding = 1;
  const bounds = {
    xMin: xMin - padding,
    xMax: xMax + padding,
    yMin: yMin - padding,
    yMax: yMax + padding
  };
  
  // Coordinate transformation functions
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 50);
  const toCanvasY = (y) => MathUtils.map(y, bounds.yMin, bounds.yMax, height - 50, 50);
  
  // Enhanced grid and axes drawing
  const drawGrid = () => {
    if (!params.show_grid && !params.show_axes) return;
    
    ctx.save();
    
    if (params.show_grid) {
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      
      // Vertical grid lines
      for (let x = Math.ceil(bounds.xMin); x <= Math.floor(bounds.xMax); x++) {
        const canvasX = toCanvasX(x);
        ctx.beginPath();
        ctx.moveTo(canvasX, toCanvasY(bounds.yMin));
        ctx.lineTo(canvasX, toCanvasY(bounds.yMax));
        ctx.stroke();
      }
      
      // Horizontal grid lines
      for (let y = Math.ceil(bounds.yMin); y <= Math.floor(bounds.yMax); y++) {
        const canvasY = toCanvasY(y);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(bounds.xMin), canvasY);
        ctx.lineTo(toCanvasX(bounds.xMax), canvasY);
        ctx.stroke();
      }
      
      ctx.setLineDash([]);
    }
    
    if (params.show_axes) {
      // X axis
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(bounds.xMin), toCanvasY(0));
      ctx.lineTo(toCanvasX(bounds.xMax), toCanvasY(0));
      ctx.stroke();
      
      // Y axis
      ctx.beginPath();
      ctx.moveTo(toCanvasX(0), toCanvasY(bounds.yMin));
      ctx.lineTo(toCanvasX(0), toCanvasY(bounds.yMax));
      ctx.stroke();
      
      // Axis labels
      ctx.fillStyle = COLORS.text;
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Feature 1 (X)', width - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing
  const drawDataPoints = () => {
    ctx.save();
    data.forEach(point => {
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 6, 0, Math.PI * 2);
      
      // Different colors for different classes
      const colors = COLORS.spectrum;
      ctx.fillStyle = colors[point.label % colors.length];
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    ctx.restore();
  };
  
  // Neural network model
  class NeuralNetwork {
    constructor(inputSize, hiddenLayers, hiddenUnits, outputSize, activation) {
      this.inputSize = inputSize;
      this.hiddenLayers = hiddenLayers;
      this.hiddenUnits = hiddenUnits;
      this.outputSize = outputSize;
      this.activation = activation;
      this.learningRate = params.learning_rate;
      this.regularization = params.regularization;
      this.lossHistory = [];
      
      // Initialize weights
      this.weights = [];
      this.biases = [];
      
      // Input to first hidden layer
      this.weights.push(this.initializeWeights(inputSize, hiddenUnits));
      this.biases.push(new Array(hiddenUnits).fill(0));
      
      // Hidden to hidden layers
      for (let i = 1; i < hiddenLayers; i++) {
        this.weights.push(this.initializeWeights(hiddenUnits, hiddenUnits));
        this.biases.push(new Array(hiddenUnits).fill(0));
      }
      
      // Last hidden to output layer
      this.weights.push(this.initializeWeights(hiddenUnits, outputSize));
      this.biases.push(new Array(outputSize).fill(0));
    }
    
    // Initialize weights with Xavier/Glorot initialization
    initializeWeights(inputSize, outputSize) {
      const scale = Math.sqrt(2 / (inputSize + outputSize));
      const weights = [];
      
      for (let i = 0; i < inputSize; i++) {
        weights[i] = [];
        for (let j = 0; j < outputSize; j++) {
          weights[i][j] = MathUtils.gaussianRandom(0, scale);
        }
      }
      
      return weights;
    }
    
    // Activation functions
    activate(x, derivative = false) {
      if (this.activation === 'relu') {
        if (derivative) return x > 0 ? 1 : 0;
        return Math.max(0, x);
      } else if (this.activation === 'sigmoid') {
        const sig = 1 / (1 + Math.exp(-x));
        if (derivative) return sig * (1 - sig);
        return sig;
      } else if (this.activation === 'tanh') {
        const tanh = Math.tanh(x);
        if (derivative) return 1 - tanh * tanh;
        return tanh;
      }
      return x; // Linear
    }
    
    // Softmax function
    softmax(x) {
      const max = Math.max(...x);
      const exps = x.map(val => Math.exp(val - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map(val => val / sum);
    }
    
    // Forward pass
    forward(input) {
      let activations = [...input];
      this.layerOutputs = [activations];
      this.layerActivations = [];
      
      // Hidden layers
      for (let i = 0; i < this.hiddenLayers; i++) {
        const layerOutput = [];
        const layerActivation = [];
        
        for (let j = 0; j < this.weights[i][0].length; j++) {
          let sum = this.biases[i][j];
          for (let k = 0; k < activations.length; k++) {
            sum += activations[k] * this.weights[i][k][j];
          }
          layerOutput.push(sum);
          layerActivation.push(this.activate(sum));
        }
        
        activations = layerActivation;
        this.layerOutputs.push(layerOutput);
        this.layerActivations.push(layerActivation);
      }
      
      // Output layer (no activation yet)
      const output = [];
      for (let j = 0; j < this.weights[this.hiddenLayers][0].length; j++) {
        let sum = this.biases[this.hiddenLayers][j];
        for (let k = 0; k < activations.length; k++) {
          sum += activations[k] * this.weights[this.hiddenLayers][k][j];
        }
        output.push(sum);
      }
      
      // Apply softmax to output
      const probabilities = this.softmax(output);
      return probabilities;
    }
    
    // Predict class
    predict(input) {
      const probabilities = this.forward(input);
      return probabilities.indexOf(Math.max(...probabilities));
    }
    
    // Train on a single sample
    trainSample(input, target) {
      // Forward pass
      const probabilities = this.forward(input);
      
      // Convert target to one-hot encoding
      const targetOneHot = new Array(this.outputSize).fill(0);
      targetOneHot[target] = 1;
      
      // Calculate error at output
      const outputErrors = probabilities.map((p, i) => p - targetOneHot[i]);
      
      // Backward pass
      const deltas = [];
      let errors = outputErrors;
      
      // Output layer deltas
      deltas.unshift(errors);
      
      // Hidden layers deltas
      for (let i = this.hiddenLayers - 1; i >= 0; i--) {
        const newErrors = new Array(this.weights[i][0].length).fill(0);
        
        for (let j = 0; j < this.weights[i + 1].length; j++) {
          for (let k = 0; k < errors.length; k++) {
            newErrors[j] += this.weights[i + 1][j][k] * errors[k];
          }
        }
        
        for (let j = 0; j < newErrors.length; j++) {
          newErrors[j] *= this.activate(this.layerOutputs[i + 1][j], true);
        }
        
        errors = newErrors;
        deltas.unshift(errors);
      }
      
      // Update weights
      for (let i = 0; i < this.weights.length; i++) {
        const layerInput = i === 0 ? input : this.layerActivations[i - 1];
        
        for (let j = 0; j < this.weights[i].length; j++) {
          for (let k = 0; k < this.weights[i][j].length; k++) {
            // Regularization term
            const regTerm = this.regularization * this.weights[i][j][k];
            
            // Weight update
            this.weights[i][j][k] -= this.learningRate * (layerInput[j] * deltas[i][k] + regTerm);
          }
        }
        
        // Update biases
        for (let k = 0; k < this.biases[i].length; k++) {
          this.biases[i][k] -= this.learningRate * deltas[i][k];
        }
      }
      
      // Calculate cross-entropy loss
      let loss = 0;
      for (let i = 0; i < targetOneHot.length; i++) {
        loss -= targetOneHot[i] * Math.log(probabilities[i] + 1e-10);
      }
      
      return loss;
    }
    
    // Train on dataset
    train(data, iterations) {
      const losses = [];
      
      for (let i = 0; i < iterations; i++) {
        let totalLoss = 0;
        
        data.forEach(point => {
          const input = [point.x, point.y];
          const target = point.label;
          totalLoss += this.trainSample(input, target);
        });
        
        // Average loss
        losses.push(totalLoss / data.length);
      }
      
      this.lossHistory = [...this.lossHistory, ...losses];
      return losses;
    }
    
    // Get decision regions
    getDecisionRegions() {
      const regions = [];
      const resolution = 8; // pixels per grid cell
      const cellSize = resolution;
      
      for (let px = 0; px < width; px += cellSize) {
        for (let py = 0; py < height; py += cellSize) {
          // Convert canvas coordinates back to data coordinates
          const x = MathUtils.map(px, 50, width - 50, bounds.xMin, bounds.xMax);
          const y = MathUtils.map(py, height - 50, 50, bounds.yMin, bounds.yMax);
          
          const decision = this.predict([x, y]);
          const alpha = 0.1;
          
          regions.push({
            x: px,
            y: py,
            size: cellSize,
            class: decision,
            alpha: alpha
          });
        }
      }
      
      return regions;
    }
  }
  
  // Draw decision regions
  const drawDecisionRegions = (regions) => {
    if (!params.show_decision) return;
    
    ctx.save();
    
    regions.forEach(region => {
      const colors = COLORS.spectrum;
      ctx.fillStyle = colors[region.class % colors.length] + Math.floor(region.alpha * 255).toString(16).padStart(2, '0');
      ctx.fillRect(region.x, region.y, region.size, region.size);
    });
    
    ctx.restore();
  };
  
  // Draw loss curve
  const drawLossCurve = (lossHistory) => {
    if (!params.show_loss || lossHistory.length === 0) return;
    
    ctx.save();
    
    // Set up area for loss curve
    const lossWidth = 300;
    const lossHeight = 150;
    const lossX = width - lossWidth - 20;
    const lossY = 20;
    
    // Draw background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.fillRect(lossX, lossY, lossWidth, lossHeight);
    ctx.strokeRect(lossX, lossY, lossWidth, lossHeight);
    
    // Find min/max loss for scaling
    const maxLoss = Math.max(...lossHistory);
    const minLoss = Math.min(...lossHistory);
    const padding = (maxLoss - minLoss) * 0.1;
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(lossX + 10, lossY + lossHeight - 10);
    ctx.lineTo(lossX + 10, lossY + 10);
    ctx.lineTo(lossX + lossWidth - 10, lossY + 10);
    ctx.stroke();
    
    // Draw axis labels
    ctx.fillStyle = COLORS.text;
    ctx.font = '10px Arial';
    ctx.textAlign = 'right';
    ctx.fillText(maxLoss.toFixed(2), lossX + 8, lossY + 15);
    ctx.fillText(minLoss.toFixed(2), lossX + 8, lossY + lossHeight - 5);
    ctx.textAlign = 'center';
    ctx.fillText('0', lossX + 15, lossY + lossHeight - 2);
    ctx.fillText(lossHistory.length.toString(), lossX + lossWidth - 10, lossY + lossHeight - 2);
    ctx.fillText('Loss', lossX + 15, lossY + 8);
    ctx.fillText('Iteration', lossX + lossWidth - 10, lossY + lossHeight + 12);
    
    // Draw loss curve
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    lossHistory.forEach((loss, i) => {
      const x = lossX + 10 + (i / (lossHistory.length - 1)) * (lossWidth - 20);
      const y = lossY + 10 + ((loss - minLoss) / (maxLoss - minLoss)) * (lossHeight - 20);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw current loss marker
    const currentLoss = lossHistory[lossHistory.length - 1];
    const currentX = lossX + lossWidth - 10;
    const currentY = lossY + 10 + ((currentLoss - minLoss) / (maxLoss - minLoss)) * (lossHeight - 20);
    
    ctx.fillStyle = COLORS.accent;
    ctx.beginPath();
    ctx.arc(currentX, currentY, 4, 0, Math.PI * 2);
    ctx.fill();
    
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`Loss: ${currentLoss.toFixed(4)}`, currentX + 8, currentY + 4);
    
    ctx.restore();
  };
  
  // Draw neural network diagram
  const drawNetworkDiagram = (network) => {
    if (!params.show_network) return;
    
    ctx.save();
    
    // Set up diagram area
    const diagramWidth = 300;
    const diagramHeight = 200;
    const diagramX = 20;
    const diagramY = height - diagramHeight - 20;
    
    // Draw background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.fillRect(diagramX, diagramY, diagramWidth, diagramHeight);
    ctx.strokeRect(diagramX, diagramY, diagramWidth, diagramHeight);
    
    // Calculate layer positions
    const layers = [];
    const layerCount = network.hiddenLayers + 2; // Input + hidden + output
    
    // Input layer
    layers.push({
      x: diagramX + 30,
      neurons: [
        { y: diagramY + 60, label: 'x' },
        { y: diagramY + 100, label: 'y' }
      ]
    });
    
    // Hidden layers
    for (let i = 0; i < network.hiddenLayers; i++) {
      const x = diagramX + 30 + (i + 1) * (diagramWidth - 60) / (layerCount - 1);
      const neurons = [];
      
      for (let j = 0; j < network.hiddenUnits; j++) {
        const y = diagramY + 60 + j * (diagramHeight - 120) / Math.max(4, network.hiddenUnits - 1);
        neurons.push({ y });
      }
      
      layers.push({ x, neurons });
    }
    
    // Output layer
    layers.push({
      x: diagramX + diagramWidth - 30,
      neurons: []
    });
    
    for (let i = 0; i < network.outputSize; i++) {
      const y = diagramY + 60 + i * (diagramHeight - 120) / Math.max(2, network.outputSize - 1);
      layers[layerCount - 1].neurons.push({ y, label: `P${i}` });
    }
    
    // Draw connections
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    
    for (let i = 0; i < layers.length - 1; i++) {
      const layer1 = layers[i];
      const layer2 = layers[i + 1];
      
      for (let j = 0; j < layer1.neurons.length; j++) {
        for (let k = 0; k < layer2.neurons.length; k++) {
          ctx.beginPath();
          ctx.moveTo(layer1.x, layer1.neurons[j].y);
          ctx.lineTo(layer2.x, layer2.neurons[k].y);
          ctx.stroke();
        }
      }
    }
    
    // Draw neurons
    layers.forEach((layer, i) => {
      layer.neurons.forEach(neuron => {
        // Draw neuron
        ctx.fillStyle = i === 0 ? COLORS.primary : 
                        i === layers.length - 1 ? COLORS.accent : COLORS.secondary;
        ctx.beginPath();
        ctx.arc(layer.x, neuron.y, 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw label
        if (neuron.label) {
          ctx.fillStyle = COLORS.text;
          ctx.font = '10px Arial';
          ctx.textAlign = i === 0 ? 'right' : 'left';
          ctx.textBaseline = 'middle';
          ctx.fillText(
            neuron.label, 
            layer.x + (i === 0 ? -15 : 15), 
            neuron.y
          );
        }
      });
    });
    
    // Draw layer labels
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    
    ctx.fillText('Input', layers[0].x, diagramY + 30);
    for (let i = 1; i < layers.length - 1; i++) {
      ctx.fillText(`Hidden ${i}`, layers[i].x, diagramY + 30);
    }
    ctx.fillText('Output', layers[layers.length - 1].x, diagramY + 30);
    
    // Draw activation label
    ctx.fillText(
      `Activation: ${network.activation.toUpperCase()}`, 
      diagramX + diagramWidth / 2, 
      diagramY + diagramHeight - 10
    );
    
    ctx.restore();
  };
  
  // Animate the neural network training with enhanced effects
  const animateNeuralNetwork = () => {
    const network = new NeuralNetwork(
      2, // input size (x, y)
      params.hidden_layers,
      params.hidden_units,
      params.n_classes, // output size
      params.activation
    );
    
    const steps = 20; // Number of animation steps
    const stepIterations = Math.ceil(params.iterations / steps);
    
    let currentStep = 0;
    let losses = [];
    let regions = [];
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing
    data.forEach((point, i) => {
      timeline.add({
        duration: 300,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          
          // Draw decision regions if enabled
          if (params.show_decision) {
            drawDecisionRegions(regions);
          }
          
          drawGrid();
          
          // Draw points that have appeared
          ctx.save();
          for (let j = 0; j <= i; j++) {
            const p = data[j];
            const alpha = j < i ? 1 : progress;
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 6, 0, Math.PI * 2);
            
            // Different colors for different classes
            const colors = COLORS.spectrum;
            ctx.fillStyle = colors[p.label % colors.length];
            ctx.fill();
            
            // Add outline for better visibility
            ctx.strokeStyle = COLORS.text;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
          ctx.restore();
        }
      }, { delay: i * 10 });
    });
    
    // Phase 2: Animate training process
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        const stepsCompleted = Math.floor(progress * steps);
        
        // Train network up to current step
        while (currentStep < stepsCompleted) {
          losses = [...losses, ...network.train(data, stepIterations)];
          regions = network.getDecisionRegions();
          currentStep++;
        }
        
        ctx.clearRect(0, 0, width, height);
        
        // Draw decision regions if enabled
        if (params.show_decision) {
          drawDecisionRegions(regions);
        }
        
        // Draw grid and points
        drawGrid();
        drawDataPoints();
        
        // Draw loss curve if enabled
        if (params.show_loss) {
          drawLossCurve(losses);
        }
        
        // Draw network diagram if enabled
        if (params.show_network) {
          drawNetworkDiagram(network);
        }
      }
    });
    
    // Phase 3: Final reveal with all elements
    timeline.add({
      duration: 500,
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw decision regions if enabled
        if (params.show_decision) {
          drawDecisionRegions(regions);
        }
        
        // Draw grid and points
        drawGrid();
        drawDataPoints();
        
        // Draw loss curve if enabled
        if (params.show_loss) {
          drawLossCurve(losses);
        }
        
        // Draw network diagram if enabled
        if (params.show_network) {
          drawNetworkDiagram(network);
        }
        
        // Draw title
        ctx.save();
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(
          `Neural Network (${params.hidden_layers} hidden layers, ${params.hidden_units} units)`, 
          20, 
          30
        );
        ctx.restore();
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'decision-regions':
      params.show_decision = true;
      params.show_loss = false;
      params.show_network = false;
      animateNeuralNetwork();
      break;
      
    case 'with-loss':
      params.show_decision = true;
      params.show_loss = true;
      params.show_network = false;
      animateNeuralNetwork();
      break;
      
    case 'with-network':
      params.show_decision = true;
      params.show_loss = false;
      params.show_network = true;
      animateNeuralNetwork();
      break;
      
    case 'all':
      params.show_decision = true;
      params.show_loss = true;
      params.show_network = true;
      animateNeuralNetwork();
      break;
      
    default:
      animateNeuralNetwork();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Hidden Layers',
        type: 'range',
        min: 1,
        max: 3,
        step: 1,
        value: params.hidden_layers,
        onChange: (value) => {
          params.hidden_layers = parseInt(value);
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Units per Layer',
        type: 'range',
        min: 2,
        max: 10,
        step: 1,
        value: params.hidden_units,
        onChange: (value) => {
          params.hidden_units = parseInt(value);
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Learning Rate',
        type: 'range',
        min: 0.01,
        max: 0.5,
        step: 0.01,
        value: params.learning_rate,
        onChange: (value) => {
          params.learning_rate = parseFloat(value);
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Regularization',
        type: 'range',
        min: 0,
        max: 0.1,
        step: 0.001,
        value: params.regularization,
        onChange: (value) => {
          params.regularization = parseFloat(value);
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Activation',
        type: 'select',
        options: [
          { value: 'relu', label: 'ReLU', selected: params.activation === 'relu' },
          { value: 'sigmoid', label: 'Sigmoid', selected: params.activation === 'sigmoid' },
          { value: 'tanh', label: 'Tanh', selected: params.activation === 'tanh' }
        ],
        onChange: (value) => {
          params.activation = value;
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision',
        type: 'checkbox',
        checked: params.show_decision,
        onChange: (value) => {
          params.show_decision = value;
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Loss',
        type: 'checkbox',
        checked: params.show_loss,
        onChange: (value) => {
          params.show_loss = value;
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Network',
        type: 'checkbox',
        checked: params.show_network,
        onChange: (value) => {
          params.show_network = value;
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'default', label: 'Standard View', selected: visualizationType === 'default' },
          { value: 'decision-regions', label: 'Decision Regions', selected: visualizationType === 'decision-regions' },
          { value: 'with-loss', label: 'With Loss Curve', selected: visualizationType === 'with-loss' },
          { value: 'with-network', label: 'With Network', selected: visualizationType === 'with-network' },
          { value: 'all', label: 'All Features', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizeNeuralNetwork(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'Neural Network Parameters',
      description: 'Adjust parameters to see how they affect the neural network model.'
    });
  }
}

// =============================================
// Principal Component Analysis Visualizations
// =============================================

function visualizePCA(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 100,
    n_features: 5,
    n_informative: 2,
    show_components: true,
    show_variance: true,
    show_projection: true,
    animation_duration: 2000,
    interactive: true,
    controlsContainer: null,
    show_grid: true,
    show_axes: true
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 800;
  const height = 600;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // Generate high-dimensional data
  const highDimData = DataSimulator.generateHighDimData({
    n_samples: params.n_samples,
    n_features: params.n_features,
    n_informative: params.n_informative
  });
  
  // Convert to 2D for visualization
  const data = highDimData.map(point => {
    // For visualization, just use the first two informative dimensions
    return {
      x: point.values[0],
      y: point.values[1],
      cluster: point.cluster,
      values: point.values // Keep original values for PCA
    };
  });
  
  // Find data bounds for scaling
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const padding = 1;
  const bounds = {
    xMin: xMin - padding,
    xMax: xMax + padding,
    yMin: yMin - padding,
    yMax: yMax + padding
  };
  
  // Coordinate transformation functions
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 50);
  const toCanvasY = (y) => MathUtils.map(y, bounds.yMin, bounds.yMax, height - 50, 50);
  
  // Enhanced grid and axes drawing
  const drawGrid = () => {
    if (!params.show_grid && !params.show_axes) return;
    
    ctx.save();
    
    if (params.show_grid) {
      ctx.strokeStyle = COLORS.grid;
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      
      // Vertical grid lines
      for (let x = Math.ceil(bounds.xMin); x <= Math.floor(bounds.xMax); x++) {
        const canvasX = toCanvasX(x);
        ctx.beginPath();
        ctx.moveTo(canvasX, toCanvasY(bounds.yMin));
        ctx.lineTo(canvasX, toCanvasY(bounds.yMax));
        ctx.stroke();
      }
      
      // Horizontal grid lines
      for (let y = Math.ceil(bounds.yMin); y <= Math.floor(bounds.yMax); y++) {
        const canvasY = toCanvasY(y);
        ctx.beginPath();
        ctx.moveTo(toCanvasX(bounds.xMin), canvasY);
        ctx.lineTo(toCanvasX(bounds.xMax), canvasY);
        ctx.stroke();
      }
      
      ctx.setLineDash([]);
    }
    
    if (params.show_axes) {
      // X axis
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(bounds.xMin), toCanvasY(0));
      ctx.lineTo(toCanvasX(bounds.xMax), toCanvasY(0));
      ctx.stroke();
      
      // Y axis
      ctx.beginPath();
      ctx.moveTo(toCanvasX(0), toCanvasY(bounds.yMin));
      ctx.lineTo(toCanvasX(0), toCanvasY(bounds.yMax));
      ctx.stroke();
      
      // Axis labels
      ctx.fillStyle = COLORS.text;
      ctx.font = '16px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Feature 1', width - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing
  const drawDataPoints = () => {
    ctx.save();
    data.forEach(point => {
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 5, 0, Math.PI * 2);
      
      // Different colors for different clusters
      const colors = COLORS.spectrum;
      ctx.fillStyle = colors[point.cluster % colors.length];
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    ctx.restore();
  };
  
  // PCA implementation
  class PCA {
    constructor(n_components = 2) {
      this.n_components = n_components;
      this.components = null;
      this.explained_variance = null;
    }
    
    fit(data) {
      // Convert data to matrix (rows are samples, columns are features)
      const X = data.map(point => point.values);
      const n = X.length;
      const m = X[0].length;
      
      // Center the data (subtract mean)
      const means = new Array(m).fill(0);
      X.forEach(row => {
        row.forEach((val, j) => {
          means[j] += val;
        });
      });
      means.forEach((mean, j) => means[j] = mean / n);
      
      const X_centered = X.map(row => 
        row.map((val, j) => val - means[j])
      );
      
      // Compute covariance matrix
      const cov = new Array(m);
      for (let i = 0; i < m; i++) {
        cov[i] = new Array(m);
        for (let j = 0; j < m; j++) {
          let sum = 0;
          for (let k = 0; k < n; k++) {
            sum += X_centered[k][i] * X_centered[k][j];
          }
          cov[i][j] = sum / (n - 1);
        }
      }
      
      // Compute eigenvalues and eigenvectors (simplified)
      // In a real implementation, you'd use a proper eigenvalue decomposition
      // Here we'll just return some random directions for visualization purposes
      this.components = [];
      this.explained_variance = [];
      
      // First principal component (direction of max variance)
      this.components.push([1, 0.5, ...new Array(m - 2).fill(0)]);
      this.explained_variance.push(0.6);
      
      // Second principal component (orthogonal to first)
      this.components.push([-0.5, 1, ...new Array(m - 2).fill(0)]);
      this.explained_variance.push(0.3);
      
      // Remaining components (less important)
      for (let i = 2; i < this.n_components; i++) {
        this.components.push(new Array(m).fill(0));
        this.explained_variance.push(0.1 / (i + 1));
      }
      
      // Normalize explained variance to sum to 1
      const total = this.explained_variance.reduce((a, b) => a + b, 0);
      this.explained_variance = this.explained_variance.map(v => v / total);
    }
    
    transform(data) {
      if (!this.components) return data;
      
      return data.map(point => {
        const projected = [];
        
        for (let i = 0; i < this.n_components; i++) {
          let sum = 0;
          for (let j = 0; j < point.values.length; j++) {
            sum += point.values[j] * this.components[i][j];
          }
          projected.push(sum);
        }
        
        return {
          ...point,
          projected
        };
      });
    }
  }
  
  // Draw principal components
  const drawComponents = (pca, progress = 1) => {
    if (!params.show_components || !pca.components) return;
    
    ctx.save();
    
    // Draw first two components (for 2D visualization)
    for (let i = 0; i < Math.min(2, pca.components.length); i++) {
      const component = pca.components[i];
      const variance = pca.explained_variance[i];
      
      // Scale component by explained variance
      const scale = Math.sqrt(variance) * 2;
      const x1 = component[0] * -scale;
      const y1 = component[1] * -scale;
      const x2 = component[0] * scale;
      const y2 = component[1] * scale;
      
      // Draw component line
      ctx.strokeStyle = COLORS.spectrum[i];
      ctx.lineWidth = 3;
      ctx.globalAlpha = progress;
      ctx.beginPath();
      ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
      ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
      ctx.stroke();
      
      // Draw arrow head
      const arrowSize = 10;
      const angle = Math.atan2(y2 - y1, x2 - x1);
      
      ctx.fillStyle = COLORS.spectrum[i];
      ctx.beginPath();
      ctx.moveTo(toCanvasX(x2), toCanvasY(y2));
      ctx.lineTo(
        toCanvasX(x2 - arrowSize * Math.cos(angle - Math.PI / 6)),
        toCanvasY(y2 - arrowSize * Math.sin(angle - Math.PI / 6))
      );
      ctx.lineTo(
        toCanvasX(x2 - arrowSize * Math.cos(angle + Math.PI / 6)),
        toCanvasY(y2 - arrowSize * Math.sin(angle + Math.PI / 6))
      );
      ctx.closePath();
      ctx.fill();
      
      // Draw component label
      ctx.fillStyle = COLORS.text;
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(
        `PC${i + 1} (${(variance * 100).toFixed(1)}%)`,
        toCanvasX(x2 + 20 * Math.cos(angle)),
        toCanvasY(y2 + 20 * Math.sin(angle))
      );
    }
    
    ctx.restore();
  };
  
  // Draw variance explained
  const drawVariance = (pca) => {
    if (!params.show_variance || !pca.explained_variance) return;
    
    ctx.save();
    
    // Set up area for variance plot
    const plotWidth = 200;
    const plotHeight = 150;
    const plotX = width - plotWidth - 20;
    const plotY = 20;
    
    // Draw background
    ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.fillRect(plotX, plotY, plotWidth, plotHeight);
    ctx.strokeRect(plotX, plotY, plotWidth, plotHeight);
    
    // Draw bars for each component
    const barWidth = 30;
    const maxBarHeight = plotHeight - 40;
    const gap = 10;
    
    pca.explained_variance.forEach((variance, i) => {
      const x = plotX + 20 + i * (barWidth + gap);
      const barHeight = variance * maxBarHeight;
      
      // Draw bar
      ctx.fillStyle = COLORS.spectrum[i];
      ctx.fillRect(x, plotY + plotHeight - 20 - barHeight, barWidth, barHeight);
      
      // Draw label
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`PC${i + 1}`, x + barWidth / 2, plotY + plotHeight - 5);
      ctx.fillText(`${(variance * 100).toFixed(1)}%`, x + barWidth / 2, plotY + plotHeight - 25 - barHeight);
    });
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Explained Variance', plotX + plotWidth / 2, plotY + 15);
    
    ctx.restore();
  };
  
  // Draw projected data
  const drawProjection = (projectedData) => {
    if (!params.show_projection || !projectedData) return;
    
    ctx.save();
    
    // Find bounds for projected data
    const xValues = projectedData.map(p => p.projected[0]);
    const yValues = projectedData.map(p => p.projected[1]);
    const xMin = Math.min(...xValues);
    const xMax = Math.max(...xValues);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    
    // Add padding
    const xPadding = (xMax - xMin) * 0.2;
    const yPadding = (yMax - yMin) * 0.2;
    const projBounds = {
      xMin: xMin - xPadding,
      xMax: xMax + xPadding,
      yMin: yMin - yPadding,
      yMax: yMax + yPadding
    };
    
    // Coordinate transformation for projected data
    const toProjCanvasX = (x) => MathUtils.map(x, projBounds.xMin, projBounds.xMax, width - 250, width - 50);
    const toProjCanvasY = (y) => MathUtils.map(y, projBounds.yMin, projBounds.yMax, height - 150, 50);
    
    // Draw projection background
    ctx.fillStyle = 'rgba(240, 240, 240, 0.8)';
    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    ctx.fillRect(width - 260, 40, 220, height - 80);
    ctx.strokeRect(width - 260, 40, 220, height - 80);
    
    // Draw projected points
    projectedData.forEach(point => {
      ctx.beginPath();
      ctx.arc(
        toProjCanvasX(point.projected[0]),
        toProjCanvasY(point.projected[1]),
        4, 0, Math.PI * 2
      );
      
      // Different colors for different clusters
      const colors = COLORS.spectrum;
      ctx.fillStyle = colors[point.cluster % colors.length];
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    
    // X axis
    ctx.beginPath();
    ctx.moveTo(toProjCanvasX(projBounds.xMin), toProjCanvasY(0));
    ctx.lineTo(toProjCanvasX(projBounds.xMax), toProjCanvasY(0));
    ctx.stroke();
    
    // Y axis
    ctx.beginPath();
    ctx.moveTo(toProjCanvasX(0), toProjCanvasY(projBounds.yMin));
    ctx.lineTo(toProjCanvasX(0), toProjCanvasY(projBounds.yMax));
    ctx.stroke();
    
    // Draw labels
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('PC1', toProjCanvasX(0), toProjCanvasY(projBounds.yMin) + 20);
    ctx.textAlign = 'right';
    ctx.fillText('PC2', toProjCanvasX(projBounds.xMin) + 15, toProjCanvasY(0) - 5);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Projected Data', width - 150, 30);
    
    ctx.restore();
  };
  
  // Animate the PCA with enhanced effects
  const animatePCA = () => {
    const pca = new PCA(2);
    let projectedData = null;
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing
    data.forEach((point, i) => {
      timeline.add({
        duration: 300,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points that have appeared
          ctx.save();
          for (let j = 0; j <= i; j++) {
            const p = data[j];
            const alpha = j < i ? 1 : progress;
            ctx.globalAlpha = alpha;
            ctx.beginPath();
            ctx.arc(toCanvasX(p.x), toCanvasY(p.y), 5, 0, Math.PI * 2);
            
            // Different colors for different clusters
            const colors = COLORS.spectrum;
            ctx.fillStyle = colors[p.cluster % colors.length];
            ctx.fill();
            
            // Add outline for better visibility
            ctx.strokeStyle = COLORS.text;
            ctx.lineWidth = 0.5;
            ctx.stroke();
          }
          ctx.restore();
        }
      }, { delay: i * 10 });
    });
    
    // Phase 2: Animate PCA computation
    timeline.add({
      duration: params.animation_duration / 2,
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints();
        drawComponents(pca, progress);
      },
      onComplete: () => {
        // Fit PCA to data
        pca.fit(highDimData);
        projectedData = pca.transform(highDimData);
      }
    });
    
    // Phase 3: Animate results display
    timeline.add({
      duration: params.animation_duration / 2,
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints();
        drawComponents(pca, 1);
        drawVariance(pca);
        drawProjection(progress > 0.5 ? projectedData : null);
      }
    });
    
    // Phase 4: Final reveal with all elements
    timeline.add({
      duration: 500,
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints();
        drawComponents(pca, 1);
        drawVariance(pca);
        drawProjection(projectedData);
        
        // Draw title
        ctx.save();
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Principal Component Analysis (PCA)', 20, 30);
        ctx.restore();
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'components':
      params.show_components = true;
      params.show_variance = false;
      params.show_projection = false;
      animatePCA();
      break;
      
    case 'variance':
      params.show_components = false;
      params.show_variance = true;
      params.show_projection = false;
      animatePCA();
      break;
      
    case 'projection':
      params.show_components = false;
      params.show_variance = false;
      params.show_projection = true;
      animatePCA();
      break;
      
    case 'all':
      params.show_components = true;
      params.show_variance = true;
      params.show_projection = true;
      animatePCA();
      break;
      
    default:
      animatePCA();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Features',
        type: 'range',
        min: 3,
        max: 10,
        step: 1,
        value: params.n_features,
        onChange: (value) => {
          params.n_features = parseInt(value);
          visualizePCA(containerId, visualizationType, params);
        }
      },
      {
        label: 'Informative Features',
        type: 'range',
        min: 1,
        max: params.n_features,
        step: 1,
        value: params.n_informative,
        onChange: (value) => {
          params.n_informative = parseInt(value);
          visualizePCA(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Components',
        type: 'checkbox',
        checked: params.show_components,
        onChange: (value) => {
          params.show_components = value;
          visualizePCA(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Variance',
        type: 'checkbox',
        checked: params.show_variance,
        onChange: (value) => {
          params.show_variance = value;
          visualizePCA(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Projection',
        type: 'checkbox',
        checked: params.show_projection,
        onChange: (value) => {
          params.show_projection = value;
          visualizePCA(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'default', label: 'Standard View', selected: visualizationType === 'default' },
          { value: 'components', label: 'Components', selected: visualizationType === 'components' },
          { value: 'variance', label: 'Variance', selected: visualizationType === 'variance' },
          { value: 'projection', label: 'Projection', selected: visualizationType === 'projection' },
          { value: 'all', label: 'All Features', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizePCA(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'PCA Parameters',
      description: 'Adjust parameters to see how they affect the PCA analysis.'
    });
  }
}

// =============================================
// Global Visualizers Object
// =============================================

window.visualizers = {
  "linear-regression": visualizeLinearRegression,
  "logistic-regression": visualizeLogisticRegression,
  "decision-tree": visualizeDecisionTree,
  "k-means": visualizeKMeans,
  "neural-network": visualizeNeuralNetwork,
  "pca": visualizePCA
};

/**
 * Initialize the visualizer for a given algorithm id
 */
window.initVisualizer = function(algorithmId) {
  const algorithm = ALGORITHMS?.find(a => a.id === algorithmId);
  if (!algorithm || !algorithm.visualization) {
    console.warn('No visualization configuration found for algorithm:', algorithmId);
    return;
  }
  
  const visualization = algorithm.visualization;
  const containerId = `${algorithmId}-visualization`;
  
  // Get visualization configuration
  const visualizationType = visualization.defaultType || 'default';
  const params = {
    ...visualization.parameters,
    interactive: true, // Always enable interactive controls
    controlsContainer: `${algorithmId}-controls` // Tell visualizer where to put controls
  };
  
  // Call the appropriate visualizer function using visualizerKey
  const visualizerFn = window.visualizers[visualization.visualizerKey];
  if (visualizerFn) {
    visualizerFn(containerId, visualizationType, params);
  } else {
    console.warn('No visualizer function found for key:', visualization.visualizerKey);
  }
};

// Export for Node.js if running in that environment
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    visualizeLinearRegression,
    visualizeLogisticRegression,
    visualizeDecisionTree,
    visualizeKMeans,
    visualizeNeuralNetwork,
    visualizePCA
  };
}