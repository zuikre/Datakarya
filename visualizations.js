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
  generateDecisionTreeData: (params = {}) => {
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
    animation_duration: 3000,
    interactive: true,
    controlsContainer: null,
    trend: 'linear',
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
  
  // Enhanced data points drawing with batch animation
  const drawDataPoints = (points, progress = 1) => {
    ctx.save();
    
    // Batch draw circles for better performance
    points.forEach(point => {
      const alpha = point.outlier ? 1 : progress;
      const radius = point.outlier ? 7 : 5;
      
      ctx.beginPath();
      ctx.arc(
        toCanvasX(point.x), 
        toCanvasY(point.y), 
        radius * Math.min(progress, 1),
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
    if (params.show_equation && progress > 0.8) {
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
      
      // Animate equation appearance
      const equationAlpha = MathUtils.clamp((progress - 0.8) * 5, 0, 1);
      ctx.globalAlpha = equationAlpha;
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
  
  // Enhanced residuals drawing with animation
  const drawResiduals = (slope, intercept, progress = 1) => {
    if (!params.show_residuals) return;
    
    ctx.save();
    
    data.forEach(point => {
      const predictedY = intercept + slope * point.x;
      const canvasX = toCanvasX(point.x);
      const canvasYActual = toCanvasY(point.y);
      const canvasYPred = toCanvasY(predictedY);
      
      // Only draw residuals that are visible based on progress
      const residualProgress = MathUtils.clamp((progress - 0.7) * 3.33, 0, 1);
      if (residualProgress <= 0) return;
      
      // Draw residual line
      ctx.strokeStyle = point.outlier ? COLORS.accent : COLORS.negative;
      ctx.lineWidth = point.outlier ? 2 : 1;
      ctx.setLineDash(point.outlier ? [] : [2, 2]);
      ctx.globalAlpha = residualProgress;
      ctx.beginPath();
      ctx.moveTo(canvasX, canvasYActual);
      ctx.lineTo(canvasX, canvasYPred);
      ctx.stroke();
      
      // Highlight outliers with pulsing effect
      if (point.outlier && params.show_outliers) {
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 300);
        ctx.fillStyle = COLORS.accent + Math.floor(40 * pulse).toString(16).padStart(2, '0');
        ctx.beginPath();
        ctx.arc(canvasX, canvasYActual, 10, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw outlier label
        ctx.fillStyle = COLORS.text;
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Outlier', canvasX, canvasYActual - 15);
      }
    });
    
    ctx.restore();
  };
  
  // Animate the regression line fitting with enhanced cinematic effects
  const animateRegressionFit = () => {
    const stats = calculateRegression();
    const { slope, intercept } = stats;
    
    // Initial random line
    let currentSlope = MathUtils.random(-5, 5);
    let currentIntercept = MathUtils.random(-5, 5);
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing in batches
    const batchSize = getOptimizedParams(data.length).batchSize;
    const totalBatches = Math.ceil(data.length / batchSize);
    
    for (let batch = 0; batch < totalBatches; batch++) {
      timeline.add({
        duration: 800,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points that have appeared
          const pointsToShow = Math.min((batch + progress) * batchSize, data.length);
          drawDataPoints(data.slice(0, pointsToShow), 1);
          
          // Pulse effect for newly appearing points
          if (progress > 0.9 && batch < totalBatches - 1) {
            const newestPoint = data[Math.floor(pointsToShow) - 1];
            ctx.save();
            ctx.beginPath();
            ctx.arc(
              toCanvasX(newestPoint.x),
              toCanvasY(newestPoint.y),
              10 * (1 - (progress % 0.1) * 10),
              0, 
              Math.PI * 2
            );
            ctx.strokeStyle = COLORS.highlight;
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.restore();
          }
        }
      }, { delay: batch * 200 });
    }
    
    // Phase 2: Animate line fitting with smooth morphing
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid and points
        drawGrid();
        drawDataPoints(data, 1);
        
        // Interpolate to final line with smooth morphing
        currentSlope = MathUtils.lerp(currentSlope, slope, progress);
        currentIntercept = MathUtils.lerp(currentIntercept, intercept, progress);
        
        // Draw line and residuals
        drawRegressionLine(currentSlope, currentIntercept, progress, progress > 0.5 ? stats : null);
        
        // Show residuals in last part of animation
        if (progress > 0.7) {
          drawResiduals(currentSlope, currentIntercept, progress);
        }
      }
    });
    
    // Phase 3: Final reveal with all elements and cinematic zoom
    timeline.add({
      duration: 1000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Subtle zoom effect toward the data center
        const zoomProgress = Easing.easeOutCubic(progress);
        const centerX = (bounds.xMin + bounds.xMax) / 2;
        const centerY = (bounds.yMin + bounds.yMax) / 2;
        const zoomScale = 1 - 0.1 * zoomProgress;
        
        ctx.save();
        ctx.translate(
          toCanvasX(centerX),
          toCanvasY(centerY)
        );
        ctx.scale(zoomScale, zoomScale);
        ctx.translate(
          -toCanvasX(centerX),
          -toCanvasY(centerY)
        );
        
        drawGrid();
        drawDataPoints(data, 1);
        drawRegressionLine(slope, intercept, 1, stats);
        drawResiduals(slope, intercept, 1);
        
        // Draw equation with fade-in effect
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
          
          ctx.globalAlpha = progress;
          ctx.fillText(equationText, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 20);
          
          if (params.show_stats) {
            ctx.fillText(`R² = ${stats.rSquared.toFixed(3)}`, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 45);
            ctx.fillText(`Slope SE = ${stats.slopeStdErr.toFixed(3)}`, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 70);
            ctx.fillText(`Intercept SE = ${stats.interceptStdErr.toFixed(3)}`, toCanvasX(bounds.xMin), toCanvasY(bounds.yMax) - 95);
          }
        }
        
        ctx.restore();
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
    animation_duration: 3000,
    interactive: true,
    controlsContainer: null,
    distribution: 'linear',
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
  
  // Enhanced data points drawing with class-by-class animation
  const drawDataPoints = (points, progress = 1) => {
    ctx.save();
    
    // Group points by class for sequential animation
    const pointsByClass = {};
    points.forEach(point => {
      if (!pointsByClass[point.label]) pointsByClass[point.label] = [];
      pointsByClass[point.label].push(point);
    });
    
    // Draw each class with different timing
    Object.keys(pointsByClass).forEach((classLabel, classIndex) => {
      const classPoints = pointsByClass[classLabel];
      const classProgress = MathUtils.clamp((progress - classIndex * 0.2) * 1.25, 0, 1);
      
      classPoints.forEach(point => {
        ctx.globalAlpha = classProgress;
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
  
  // Enhanced decision boundary drawing
  const drawDecisionBoundary = (model, progress = 1) => {
    if (!params.show_boundary) return;
    
    ctx.save();
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.globalAlpha = progress;
    
    const boundary = model.getDecisionBoundary();
    ctx.beginPath();
    ctx.moveTo(toCanvasX(boundary.x1), toCanvasY(boundary.y1));
    ctx.lineTo(toCanvasX(boundary.x2), toCanvasY(boundary.y2));
    ctx.stroke();
    
    ctx.restore();
  };
  
  // Enhanced probability surface drawing
  const drawProbabilitySurface = (model, progress = 1) => {
    if (!params.show_probability) return;
    
    ctx.save();
    
    // Create a gradient for the probability surface
    const resolution = 50;
    const cellWidth = (width - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        const probability = model.predict(x, y);
        
        // Map probability to color (blue for class 0, red for class 1)
        const r = Math.floor(255 * probability);
        const g = 0;
        const b = Math.floor(255 * (1 - probability));
        
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.3 * progress})`;
        ctx.fillRect(
          toCanvasX(x) - cellWidth / 2,
          toCanvasY(y) - cellHeight / 2,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Enhanced decision regions drawing
  const drawDecisionRegions = (model, progress = 1) => {
    if (!params.show_decision) return;
    
    ctx.save();
    
    // Draw decision regions as a wave effect
    const resolution = 30;
    const cellWidth = (width - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        const predictedClass = model.predictClass(x, y);
        const color = COLORS.spectrum[predictedClass % COLORS.spectrum.length];
        
        // Wave effect based on distance from center
        const centerX = resolution / 2;
        const centerY = resolution / 2;
        const distance = Math.sqrt((i - centerX) ** 2 + (j - centerY) ** 2);
        const waveProgress = MathUtils.clamp(progress * 2 - distance / resolution, 0, 1);
        
        ctx.fillStyle = color + Math.floor(50 * waveProgress).toString(16).padStart(2, '0');
        ctx.fillRect(
          toCanvasX(x) - cellWidth / 2,
          toCanvasY(y) - cellHeight / 2,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Enhanced loss curve drawing
  const drawLossCurve = (lossHistory, progress = 1) => {
    if (!params.show_loss || lossHistory.length === 0) return;
    
    ctx.save();
    
    // Create a separate area for the loss curve
    const lossWidth = 300;
    const lossHeight = 150;
    const lossX = width - lossWidth - 20;
    const lossY = 20;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(lossX, lossY, lossWidth, lossHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(lossX, lossY, lossWidth, lossHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Loss Curve', lossX + lossWidth / 2, lossY - 5);
    
    // Find min and max loss for scaling
    const maxLoss = Math.max(...lossHistory);
    const minLoss = Math.min(...lossHistory);
    const lossRange = maxLoss - minLoss || 1; // Avoid division by zero
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(lossX + 30, lossY + 20);
    ctx.lineTo(lossX + 30, lossY + lossHeight - 20);
    ctx.lineTo(lossX + lossWidth - 20, lossY + lossHeight - 20);
    ctx.stroke();
    
    // Draw labels
    ctx.textAlign = 'right';
    ctx.fillText(minLoss.toFixed(2), lossX + 25, lossY + lossHeight - 20);
    ctx.fillText(maxLoss.toFixed(2), lossX + 25, lossY + 20);
    ctx.textAlign = 'center';
    ctx.fillText('Iteration', lossX + lossWidth / 2, lossY + lossHeight);
    
    // Draw loss curve
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const pointsToShow = Math.floor(lossHistory.length * progress);
    const visibleLosses = lossHistory.slice(0, pointsToShow);
    
    visibleLosses.forEach((loss, i) => {
      const x = lossX + 30 + (lossWidth - 50) * (i / (lossHistory.length - 1));
      const y = lossY + lossHeight - 20 - (lossHeight - 40) * ((loss - minLoss) / lossRange);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw current point
    if (pointsToShow > 0) {
      const currentLoss = lossHistory[pointsToShow - 1];
      const x = lossX + 30 + (lossWidth - 50) * ((pointsToShow - 1) / (lossHistory.length - 1));
      const y = lossY + lossHeight - 20 - (lossHeight - 40) * ((currentLoss - minLoss) / lossRange);
      
      ctx.fillStyle = COLORS.accent;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw current loss value
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(currentLoss.toFixed(4), x + 5, y);
    }
    
    ctx.restore();
  };
  
  // Animate the logistic regression training with enhanced cinematic effects
  const animateLogisticTraining = () => {
    const model = new LogisticModel();
    const iterations = params.iterations;
    const batchSize = 10; // Update in batches for smoother animation
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing class by class
    const classes = [...new Set(data.map(p => p.label))];
    
    classes.forEach((classLabel, classIndex) => {
      timeline.add({
        duration: 1000,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points from previous classes
          for (let i = 0; i < classIndex; i++) {
            const classPoints = data.filter(p => p.label === classes[i]);
            drawDataPoints(classPoints, 1);
          }
          
          // Draw points from current class with animation
          const classPoints = data.filter(p => p.label === classLabel);
          drawDataPoints(classPoints, progress);
          
          // Add class label
          if (progress > 0.8) {
            const centerX = classPoints.reduce((sum, p) => sum + p.x, 0) / classPoints.length;
            const centerY = classPoints.reduce((sum, p) => sum + p.y, 0) / classPoints.length;
            
            ctx.save();
            ctx.fillStyle = COLORS.text;
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`Class ${classLabel}`, toCanvasX(centerX), toCanvasY(centerY) - 15);
            ctx.restore();
          }
        }
      }, { delay: classIndex * 300 });
    });
    
    // Phase 2: Animate decision boundary evolution
    let currentIteration = 0;
    
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid and points
        drawGrid();
        drawDataPoints(data, 1);
        
        // Train model in batches
        const targetIteration = Math.floor(progress * iterations);
        if (targetIteration > currentIteration) {
          const iterationsToTrain = Math.min(batchSize, targetIteration - currentIteration);
          model.train(data, iterationsToTrain);
          currentIteration += iterationsToTrain;
        }
        
        // Draw decision boundary with morphing effect
        drawDecisionBoundary(model, progress);
        
        // Draw probability surface if enabled
        if (params.show_probability) {
          drawProbabilitySurface(model, progress);
        }
        
        // Draw decision regions if enabled
        if (params.show_decision) {
          drawDecisionRegions(model, progress);
        }
        
        // Draw loss curve if enabled
        if (params.show_loss) {
          drawLossCurve(model.lossHistory, progress);
        }
        
        // Draw iteration info
        ctx.fillStyle = COLORS.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Iteration: ${currentIteration}/${iterations}`, 60, 30);
        ctx.fillText(`Loss: ${model.lossHistory.length > 0 ? model.lossHistory[model.lossHistory.length - 1].toFixed(4) : 'N/A'}`, 60, 50);
      }
    });
    
    // Phase 3: Final reveal with all elements
    timeline.add({
      duration: 1000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(data, 1);
        drawDecisionBoundary(model, 1);
        
        if (params.show_probability) {
          drawProbabilitySurface(model, 1);
        }
        
        if (params.show_decision) {
          drawDecisionRegions(model, 1);
        }
        
        if (params.show_loss) {
          drawLossCurve(model.lossHistory, 1);
        }
        
        // Draw final equation
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Decision Boundary: ${model.weights[0].toFixed(2)}x + ${model.weights[1].toFixed(2)}y + ${model.bias.toFixed(2)} = 0`, 60, 70);
        ctx.fillText(`Final Loss: ${model.lossHistory[model.lossHistory.length - 1].toFixed(4)}`, 60, 90);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'decision-boundary':
      params.show_probability = false;
      params.show_decision = false;
      params.show_loss = false;
      animateLogisticTraining();
      break;
      
    case 'probability-surface':
      params.show_probability = true;
      params.show_decision = false;
      params.show_loss = false;
      animateLogisticTraining();
      break;
      
    case 'decision-regions':
      params.show_probability = false;
      params.show_decision = true;
      params.show_loss = false;
      animateLogisticTraining();
      break;
      
    case 'with-loss':
      params.show_probability = false;
      params.show_decision = false;
      params.show_loss = true;
      animateLogisticTraining();
      break;
      
    case 'all':
      params.show_probability = true;
      params.show_decision = true;
      params.show_loss = true;
      animateLogisticTraining();
      break;
      
    default:
      animateLogisticTraining();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Samples',
        type: 'range',
        min: 50,
        max: 500,
        step: 10,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizeLogisticRegression(containerId, visualizationType, params);
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
        label: 'Learning Rate',
        type: 'range',
        min: 0.01,
        max: 1,
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
          { value: 'xor', label: 'XOR', selected: params.distribution === 'xor' },
          { value: 'concentric', label: 'Concentric', selected: params.distribution === 'concentric' }
        ],
        onChange: (value) => {
          params.distribution = value;
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Number of Classes',
        type: 'select',
        options: [
          { value: 2, label: '2 Classes', selected: params.n_classes === 2 },
          { value: 3, label: '3 Classes', selected: params.n_classes === 3 },
          { value: 4, label: '4 Classes', selected: params.n_classes === 4 }
        ],
        onChange: (value) => {
          params.n_classes = parseInt(value);
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision Boundary',
        type: 'checkbox',
        checked: params.show_boundary,
        onChange: (value) => {
          params.show_boundary = value;
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Probability Surface',
        type: 'checkbox',
        checked: params.show_probability,
        onChange: (value) => {
          params.show_probability = value;
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision Regions',
        type: 'checkbox',
        checked: params.show_decision,
        onChange: (value) => {
          params.show_decision = value;
          visualizeLogisticRegression(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Loss Curve',
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
          { value: 'decision-boundary', label: 'Decision Boundary', selected: visualizationType === 'decision-boundary' },
          { value: 'probability-surface', label: 'Probability Surface', selected: visualizationType === 'probability-surface' },
          { value: 'decision-regions', label: 'Decision Regions', selected: visualizationType === 'decision-regions' },
          { value: 'with-loss', label: 'With Loss Curve', selected: visualizationType === 'with-loss' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizeLogisticRegression(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'Logistic Regression Parameters',
      description: 'Adjust parameters to see how they affect the classification model.'
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
    show_impurity: false,
    animation_duration: 3000,
    interactive: true,
    controlsContainer: null,
    distribution: 'concentric',
    show_grid: true,
    show_axes: true
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 1000;
  const height = 600;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // Generate data with more options
  const data = DataSimulator.generateDecisionTreeData({
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
  const padding = 0.5;
  const bounds = {
    xMin: xMin - padding,
    xMax: xMax + padding,
    yMin: yMin - padding,
    yMax: yMax + padding
  };
  
  // Coordinate transformation functions
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 350);
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
      ctx.fillText('Feature 1 (X)', width - 350 - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing with class-by-class animation
  const drawDataPoints = (points, progress = 1, highlight = null) => {
    ctx.save();
    
    // Group points by class for sequential animation
    const pointsByClass = {};
    points.forEach(point => {
      if (!pointsByClass[point.label]) pointsByClass[point.label] = [];
      pointsByClass[point.label].push(point);
    });
    
    // Draw each class with different timing
    Object.keys(pointsByClass).forEach((classLabel, classIndex) => {
      const classPoints = pointsByClass[classLabel];
      const classProgress = MathUtils.clamp((progress - classIndex * 0.2) * 1.25, 0, 1);
      
      classPoints.forEach(point => {
        // Dim points that are not in the current highlight
        let alpha = classProgress;
        if (highlight && !highlight.includes(point)) {
          alpha *= 0.3;
        }
        
        ctx.globalAlpha = alpha;
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
    });
    
    ctx.restore();
  };
  
  // Enhanced decision tree node class
  class TreeNode {
    constructor(data, depth = 0) {
      this.data = data;
      this.depth = depth;
      this.left = null;
      this.right = null;
      this.splitFeature = null;
      this.splitValue = null;
      this.prediction = this.getMajorityClass();
      this.impurity = this.calculateImpurity();
      this.samples = data.length;
    }
    
    getMajorityClass() {
      const classCounts = {};
      this.data.forEach(point => {
        classCounts[point.label] = (classCounts[point.label] || 0) + 1;
      });
      
      return Object.keys(classCounts).reduce((a, b) => 
        classCounts[a] > classCounts[b] ? a : b
      );
    }
    
    calculateImpurity() {
      // Gini impurity
      const classCounts = {};
      this.data.forEach(point => {
        classCounts[point.label] = (classCounts[point.label] || 0) + 1;
      });
      
      let impurity = 1;
      Object.values(classCounts).forEach(count => {
        const probability = count / this.data.length;
        impurity -= probability * probability;
      });
      
      return impurity;
    }
    
    findBestSplit() {
      let bestImpurityGain = 0;
      let bestSplit = null;
      
      // Try all possible splits on both features
      ['x', 'y'].forEach(feature => {
        // Sort data by feature value
        const sortedData = [...this.data].sort((a, b) => a[feature] - b[feature]);
        
        // Try splits between each pair of points
        for (let i = 1; i < sortedData.length; i++) {
          const splitValue = (sortedData[i-1][feature] + sortedData[i][feature]) / 2;
          
          // Split data
          const leftData = sortedData.filter(p => p[feature] <= splitValue);
          const rightData = sortedData.filter(p => p[feature] > splitValue);
          
          // Skip if split doesn't meet minimum requirements
          if (leftData.length < params.min_samples_leaf || rightData.length < params.min_samples_leaf) {
            continue;
          }
          
          // Calculate impurity gain
          const leftImpurity = new TreeNode(leftData).impurity;
          const rightImpurity = new TreeNode(rightData).impurity;
          
          const weightedImpurity = (leftData.length * leftImpurity + rightData.length * rightImpurity) / this.data.length;
          const impurityGain = this.impurity - weightedImpurity;
          
          // Update best split if this is better
          if (impurityGain > bestImpurityGain) {
            bestImpurityGain = impurityGain;
            bestSplit = {
              feature,
              value: splitValue,
              leftData,
              rightData,
              impurityGain
            };
          }
        }
      });
      
      return bestSplit;
    }
    
    buildTree() {
      // Stop if max depth reached or minimum samples not met
      if (this.depth >= params.max_depth || this.data.length < params.min_samples_split) {
        return;
      }
      
      // Find best split
      const split = this.findBestSplit();
      if (!split || split.impurityGain <= 0) {
        return;
      }
      
      // Create child nodes
      this.splitFeature = split.feature;
      this.splitValue = split.value;
      this.left = new TreeNode(split.leftData, this.depth + 1);
      this.right = new TreeNode(split.rightData, this.depth + 1);
      
      // Recursively build tree
      this.left.buildTree();
      this.right.buildTree();
    }
    
    predict(point) {
      // If leaf node, return prediction
      if (!this.left && !this.right) {
        return this.prediction;
      }
      
      // Otherwise, traverse tree
      if (point[this.splitFeature] <= this.splitValue) {
        return this.left.predict(point);
      } else {
        return this.right.predict(point);
      }
    }
  }
  
  // Enhanced decision boundary drawing
  const drawDecisionBoundaries = (tree, progress = 1) => {
    if (!params.show_splits) return;
    
    ctx.save();
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.globalAlpha = progress;
    
    // Recursively draw all splits
    const drawSplits = (node) => {
      if (!node || !node.splitFeature) return;
      
      if (node.splitFeature === 'x') {
        // Vertical split
        ctx.beginPath();
        ctx.moveTo(toCanvasX(node.splitValue), toCanvasY(bounds.yMin));
        ctx.lineTo(toCanvasX(node.splitValue), toCanvasY(bounds.yMax));
        ctx.stroke();
      } else {
        // Horizontal split
        ctx.beginPath();
        ctx.moveTo(toCanvasX(bounds.xMin), toCanvasY(node.splitValue));
        ctx.lineTo(toCanvasX(bounds.xMax), toCanvasY(node.splitValue));
        ctx.stroke();
      }
      
      // Draw split value label
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      
      if (node.splitFeature === 'x') {
        ctx.textAlign = 'center';
        ctx.fillText(node.splitValue.toFixed(2), toCanvasX(node.splitValue), toCanvasY(bounds.yMax) + 15);
      } else {
        ctx.textAlign = 'right';
        ctx.fillText(node.splitValue.toFixed(2), toCanvasX(bounds.xMin) - 5, toCanvasY(node.splitValue));
      }
      
      // Recursively draw child splits
      drawSplits(node.left);
      drawSplits(node.right);
    };
    
    drawSplits(tree);
    ctx.restore();
  };
  
  // Enhanced decision regions drawing
  const drawDecisionRegions = (tree, progress = 1) => {
    if (!params.show_regions) return;
    
    ctx.save();
    
    // Draw decision regions as a wave effect
    const resolution = 40;
    const cellWidth = (width - 350 - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        const predictedClass = tree.predict({ x, y });
        const color = COLORS.spectrum[predictedClass % COLORS.spectrum.length];
        
        // Wave effect based on distance from center
        const centerX = resolution / 2;
        const centerY = resolution / 2;
        const distance = Math.sqrt((i - centerX) ** 2 + (j - centerY) ** 2);
        const waveProgress = MathUtils.clamp(progress * 2 - distance / resolution, 0, 1);
        
        ctx.fillStyle = color + Math.floor(50 * waveProgress).toString(16).padStart(2, '0');
        ctx.fillRect(
          toCanvasX(x) - cellWidth / 2,
          toCanvasY(y) - cellHeight / 2,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Enhanced tree diagram drawing
  const drawTreeDiagram = (tree, progress = 1, currentSplit = null) => {
    if (!params.show_tree) return;
    
    ctx.save();
    
    // Tree diagram area
    const treeX = width - 320;
    const treeY = 50;
    const treeWidth = 300;
    const treeHeight = height - 100;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(treeX, treeY, treeWidth, treeHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(treeX, treeY, treeWidth, treeHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Decision Tree', treeX + treeWidth / 2, treeY - 10);
    
    // Calculate node positions
    const maxDepth = params.max_depth;
    const levelHeight = treeHeight / (maxDepth + 1);
    const nodeRadius = 20;
    
    // Recursively draw tree
    const drawNode = (node, x, y, level, isCurrent = false) => {
      if (!node) return;
      
      // Draw node
      ctx.beginPath();
      ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
      
      if (isCurrent) {
        // Highlight current node
        ctx.fillStyle = COLORS.highlight;
        ctx.fill();
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 2;
        ctx.stroke();
      } else {
        // Regular node
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      
      // Draw node content
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      
      if (node.splitFeature) {
        // Decision node
        ctx.fillText(`${node.splitFeature} ≤ ${node.splitValue.toFixed(2)}`, x, y);
        
        if (params.show_impurity && progress > 0.8) {
          ctx.font = '10px Arial';
          ctx.fillText(`Gini: ${node.impurity.toFixed(3)}`, x, y + 15);
          ctx.fillText(`Samples: ${node.samples}`, x, y + 30);
        }
      } else {
        // Leaf node
        ctx.fillText(`Class ${node.prediction}`, x, y);
        
        if (params.show_impurity && progress > 0.8) {
          ctx.font = '10px Arial';
          ctx.fillText(`Gini: ${node.impurity.toFixed(3)}`, x, y + 15);
          ctx.fillText(`Samples: ${node.samples}`, x, y + 30);
        }
      }
      
      // Draw connections to children
      if (node.left) {
        const childX = x - treeWidth / (Math.pow(2, level + 2));
        const childY = y + levelHeight;
        
        ctx.beginPath();
        ctx.moveTo(x, y + nodeRadius);
        ctx.lineTo(childX, childY - nodeRadius);
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw "Yes" label
        ctx.fillStyle = COLORS.text;
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Yes', (x + childX) / 2, (y + childY) / 2 - 5);
        
        drawNode(node.left, childX, childY, level + 1, currentSplit === node.left);
      }
      
      if (node.right) {
        const childX = x + treeWidth / (Math.pow(2, level + 2));
        const childY = y + levelHeight;
        
        ctx.beginPath();
        ctx.moveTo(x, y + nodeRadius);
        ctx.lineTo(childX, childY - nodeRadius);
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1;
        ctx.stroke();
        
        // Draw "No" label
        ctx.fillStyle = COLORS.text;
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('No', (x + childX) / 2, (y + childY) / 2 - 5);
        
        drawNode(node.right, childX, childY, level + 1, currentSplit === node.right);
      }
    };
    
    // Start drawing from root
    const rootX = treeX + treeWidth / 2;
    const rootY = treeY + levelHeight / 2;
    drawNode(tree, rootX, rootY, 0, currentSplit === tree);
    
    ctx.restore();
  };
  
  // Animate the decision tree building with enhanced cinematic effects
  const animateTreeBuilding = () => {
    const tree = new TreeNode(data);
    const splits = [];
    
    // Collect all splits in order
    const collectSplits = (node) => {
      if (!node) return;
      
      if (node.splitFeature) {
        splits.push(node);
        collectSplits(node.left);
        collectSplits(node.right);
      }
    };
    
    // Build the tree to collect splits
    tree.buildTree();
    collectSplits(tree);
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing class by class
    const classes = [...new Set(data.map(p => p.label))];
    
    classes.forEach((classLabel, classIndex) => {
      timeline.add({
        duration: 800,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points from previous classes
          for (let i = 0; i < classIndex; i++) {
            const classPoints = data.filter(p => p.label === classes[i]);
            drawDataPoints(classPoints, 1);
          }
          
          // Draw points from current class with animation
          const classPoints = data.filter(p => p.label === classLabel);
          drawDataPoints(classPoints, progress);
          
          // Add class label
          if (progress > 0.8) {
            const centerX = classPoints.reduce((sum, p) => sum + p.x, 0) / classPoints.length;
            const centerY = classPoints.reduce((sum, p) => sum + p.y, 0) / classPoints.length;
            
            ctx.save();
            ctx.fillStyle = COLORS.text;
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`Class ${classLabel}`, toCanvasX(centerX), toCanvasY(centerY) - 15);
            ctx.restore();
          }
        }
      }, { delay: classIndex * 300 });
    });
    
    // Phase 2: Animate each split with focus effect
    let currentSplitIndex = 0;
    
    splits.forEach((split, index) => {
      timeline.add({
        duration: 1500,
        easing: 'easeInOutQuad',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          
          // Draw grid
          drawGrid();
          
          // Highlight current split points
          const pointsToHighlight = split.data;
          drawDataPoints(data, 1, pointsToHighlight);
          
          // Draw splits up to current one
          for (let i = 0; i < index; i++) {
            const prevSplit = splits[i];
            ctx.save();
            ctx.strokeStyle = COLORS.accent;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.5;
            
            if (prevSplit.splitFeature === 'x') {
              ctx.beginPath();
              ctx.moveTo(toCanvasX(prevSplit.splitValue), toCanvasY(bounds.yMin));
              ctx.lineTo(toCanvasX(prevSplit.splitValue), toCanvasY(bounds.yMax));
              ctx.stroke();
            } else {
              ctx.beginPath();
              ctx.moveTo(toCanvasX(bounds.xMin), toCanvasY(prevSplit.splitValue));
              ctx.lineTo(toCanvasX(bounds.xMax), toCanvasY(prevSplit.splitValue));
              ctx.stroke();
            }
            
            ctx.restore();
          }
          
          // Draw current split with animation
          ctx.save();
          ctx.strokeStyle = COLORS.accent;
          ctx.lineWidth = 2 + 2 * progress;
          ctx.globalAlpha = progress;
          
          if (split.splitFeature === 'x') {
            ctx.beginPath();
            ctx.moveTo(toCanvasX(split.splitValue), toCanvasY(bounds.yMin));
            ctx.lineTo(toCanvasX(split.splitValue), toCanvasY(bounds.yMax));
            ctx.stroke();
          } else {
            ctx.beginPath();
            ctx.moveTo(toCanvasX(bounds.xMin), toCanvasY(split.splitValue));
            ctx.lineTo(toCanvasX(bounds.xMax), toCanvasY(split.splitValue));
            ctx.stroke();
          }
          
          ctx.restore();
          
          // Draw tree diagram up to current split
          drawTreeDiagram(tree, progress, split);
          
          // Draw split info
          ctx.fillStyle = COLORS.text;
          ctx.font = '14px Arial';
          ctx.textAlign = 'left';
          ctx.fillText(`Split ${index + 1}/${splits.length}: ${split.splitFeature} ≤ ${split.splitValue.toFixed(2)}`, 60, 30);
          ctx.fillText(`Gini Impurity: ${split.impurity.toFixed(3)}`, 60, 50);
          ctx.fillText(`Samples: ${split.samples}`, 60, 70);
        }
      }, { delay: index * 500 });
    });
    
    // Phase 3: Final reveal with all elements
    timeline.add({
      duration: 2000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(data, 1);
        drawDecisionBoundaries(tree, progress);
        drawDecisionRegions(tree, progress);
        drawTreeDiagram(tree, progress);
        
        // Draw final stats
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Final Tree Depth: ${params.max_depth}`, 60, 30);
        ctx.fillText(`Total Splits: ${splits.length}`, 60, 50);
        ctx.fillText(`Training Accuracy: ${calculateAccuracy(tree, data).toFixed(2)}%`, 60, 70);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Helper function to calculate accuracy
  const calculateAccuracy = (tree, data) => {
    let correct = 0;
    data.forEach(point => {
      if (tree.predict(point) == point.label) {
        correct++;
      }
    });
    return (correct / data.length) * 100;
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'tree-split':
      params.show_regions = false;
      params.show_impurity = false;
      animateTreeBuilding();
      break;
      
    case 'decision-boundary':
      params.show_tree = false;
      params.show_impurity = false;
      animateTreeBuilding();
      break;
      
    case 'tree-diagram':
      params.show_splits = false;
      params.show_regions = false;
      params.show_impurity = false;
      animateTreeBuilding();
      break;
      
    case 'all':
      params.show_tree = true;
      params.show_splits = true;
      params.show_regions = true;
      params.show_impurity = true;
      animateTreeBuilding();
      break;
      
    case 'with-impurity':
      params.show_tree = true;
      params.show_splits = true;
      params.show_regions = true;
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
        label: 'Number of Samples',
        type: 'range',
        min: 50,
        max: 500,
        step: 10,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Number of Classes',
        type: 'select',
        options: [
          { value: 2, label: '2 Classes', selected: params.n_classes === 2 },
          { value: 3, label: '3 Classes', selected: params.n_classes === 3 },
          { value: 4, label: '4 Classes', selected: params.n_classes === 4 }
        ],
        onChange: (value) => {
          params.n_classes = parseInt(value);
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Max Tree Depth',
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
        label: 'Data Distribution',
        type: 'select',
        options: [
          { value: 'concentric', label: 'Concentric', selected: params.distribution === 'concentric' },
          { value: 'checkerboard', label: 'Checkerboard', selected: params.distribution === 'checkerboard' },
          { value: 'linear', label: 'Linear', selected: params.distribution === 'linear' },
          { value: 'random', label: 'Random', selected: params.distribution === 'random' }
        ],
        onChange: (value) => {
          params.distribution = value;
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Tree Diagram',
        type: 'checkbox',
        checked: params.show_tree,
        onChange: (value) => {
          params.show_tree = value;
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision Boundaries',
        type: 'checkbox',
        checked: params.show_splits,
        onChange: (value) => {
          params.show_splits = value;
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision Regions',
        type: 'checkbox',
        checked: params.show_regions,
        onChange: (value) => {
          params.show_regions = value;
          visualizeDecisionTree(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Impurity Metrics',
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
          { value: 'tree-split', label: 'Splitting Process', selected: visualizationType === 'tree-split' },
          { value: 'decision-boundary', label: 'Decision Boundaries', selected: visualizationType === 'decision-boundary' },
          { value: 'tree-diagram', label: 'Tree Diagram', selected: visualizationType === 'tree-diagram' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' },
          { value: 'with-impurity', label: 'With Metrics', selected: visualizationType === 'with-impurity' }
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
// Enhanced Random Forest Visualizations
// =============================================
function visualizeRandomForest(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_trees: 10,
    max_depth: 4,
    n_samples: 200,
    n_classes: 3,
    bootstrap_ratio: 0.8,
    feature_subsampling: 0.7,
    show_individual_trees: false,
    show_ensemble: true,
    show_importance: false,
    show_oob: false,
    animation_duration: 2500,
    interactive: true,
    controlsContainer: null,
    distribution: 'concentric',
    show_grid: true,
    show_axes: true
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 1000;
  const height = 600;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // Generate data with more options
  const data = DataSimulator.generateDecisionTreeData({
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
  const padding = 0.5;
  const bounds = {
    xMin: xMin - padding,
    xMax: xMax + padding,
    yMin: yMin - padding,
    yMax: yMax + padding
  };
  
  // Coordinate transformation functions
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 350);
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
      ctx.fillText('Feature 1 (X)', width - 350 - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing with class-by-class animation
  const drawDataPoints = (points, progress = 1, highlight = null) => {
    ctx.save();
    
    // Group points by class for sequential animation
    const pointsByClass = {};
    points.forEach(point => {
      if (!pointsByClass[point.label]) pointsByClass[point.label] = [];
      pointsByClass[point.label].push(point);
    });
    
    // Draw each class with different timing
    Object.keys(pointsByClass).forEach((classLabel, classIndex) => {
      const classPoints = pointsByClass[classLabel];
      const classProgress = MathUtils.clamp((progress - classIndex * 0.2) * 1.25, 0, 1);
      
      classPoints.forEach(point => {
        // Dim points that are not in the current highlight
        let alpha = classProgress;
        if (highlight && !highlight.includes(point)) {
          alpha *= 0.3;
        }
        
        ctx.globalAlpha = alpha;
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
    });
    
    ctx.restore();
  };
  
  // Enhanced decision tree node class (same as in decision tree visualization)
  class TreeNode {
    constructor(data, depth = 0) {
      this.data = data;
      this.depth = depth;
      this.left = null;
      this.right = null;
      this.splitFeature = null;
      this.splitValue = null;
      this.prediction = this.getMajorityClass();
      this.impurity = this.calculateImpurity();
      this.samples = data.length;
    }
    
    getMajorityClass() {
      const classCounts = {};
      this.data.forEach(point => {
        classCounts[point.label] = (classCounts[point.label] || 0) + 1;
      });
      
      return Object.keys(classCounts).reduce((a, b) => 
        classCounts[a] > classCounts[b] ? a : b
      );
    }
    
    calculateImpurity() {
      // Gini impurity
      const classCounts = {};
      this.data.forEach(point => {
        classCounts[point.label] = (classCounts[point.label] || 0) + 1;
      });
      
      let impurity = 1;
      Object.values(classCounts).forEach(count => {
        const probability = count / this.data.length;
        impurity -= probability * probability;
      });
      
      return impurity;
    }
    
    findBestSplit() {
      let bestImpurityGain = 0;
      let bestSplit = null;
      
      // Try all possible splits on both features
      ['x', 'y'].forEach(feature => {
        // Sort data by feature value
        const sortedData = [...this.data].sort((a, b) => a[feature] - b[feature]);
        
        // Try splits between each pair of points
        for (let i = 1; i < sortedData.length; i++) {
          const splitValue = (sortedData[i-1][feature] + sortedData[i][feature]) / 2;
          
          // Split data
          const leftData = sortedData.filter(p => p[feature] <= splitValue);
          const rightData = sortedData.filter(p => p[feature] > splitValue);
          
          // Skip if split doesn't meet minimum requirements
          if (leftData.length < 1 || rightData.length < 1) {
            continue;
          }
          
          // Calculate impurity gain
          const leftImpurity = new TreeNode(leftData).impurity;
          const rightImpurity = new TreeNode(rightData).impurity;
          
          const weightedImpurity = (leftData.length * leftImpurity + rightData.length * rightImpurity) / this.data.length;
          const impurityGain = this.impurity - weightedImpurity;
          
          // Update best split if this is better
          if (impurityGain > bestImpurityGain) {
            bestImpurityGain = impurityGain;
            bestSplit = {
              feature,
              value: splitValue,
              leftData,
              rightData,
              impurityGain
            };
          }
        }
      });
      
      return bestSplit;
    }
    
    buildTree(maxDepth) {
      // Stop if max depth reached or minimum samples not met
      if (this.depth >= maxDepth || this.data.length < 2) {
        return;
      }
      
      // Find best split
      const split = this.findBestSplit();
      if (!split || split.impurityGain <= 0) {
        return;
      }
      
      // Create child nodes
      this.splitFeature = split.feature;
      this.splitValue = split.value;
      this.left = new TreeNode(split.leftData, this.depth + 1);
      this.right = new TreeNode(split.rightData, this.depth + 1);
      
      // Recursively build tree
      this.left.buildTree(maxDepth);
      this.right.buildTree(maxDepth);
    }
    
    predict(point) {
      // If leaf node, return prediction
      if (!this.left && !this.right) {
        return this.prediction;
      }
      
      // Otherwise, traverse tree
      if (point[this.splitFeature] <= this.splitValue) {
        return this.left.predict(point);
      } else {
        return this.right.predict(point);
      }
    }
  }
  
  // Random Forest class
  class RandomForest {
    constructor() {
      this.trees = [];
      this.oobErrors = [];
      this.featureImportance = { x: 0, y: 0 };
    }
    
    train(data, nTrees, maxDepth, bootstrapRatio) {
      this.trees = [];
      this.oobErrors = [];
      
      for (let i = 0; i < nTrees; i++) {
        // Create bootstrap sample
        const bootstrapSample = [];
        const oobSample = [];
        const usedIndices = new Set();
        
        for (let j = 0; j < data.length * bootstrapRatio; j++) {
          const randomIndex = Math.floor(Math.random() * data.length);
          usedIndices.add(randomIndex);
          bootstrapSample.push(data[randomIndex]);
        }
        
        // Create out-of-bag sample
        for (let j = 0; j < data.length; j++) {
          if (!usedIndices.has(j)) {
            oobSample.push(data[j]);
          }
        }
        
        // Build tree
        const tree = new TreeNode(bootstrapSample);
        tree.buildTree(maxDepth);
        this.trees.push(tree);
        
        // Calculate OOB error
        if (oobSample.length > 0) {
          let correct = 0;
          oobSample.forEach(point => {
            const prediction = this.predict(point, i + 1); // Predict with trees built so far
            if (prediction == point.label) {
              correct++;
            }
          });
          this.oobErrors.push(1 - (correct / oobSample.length));
        }
        
        // Update feature importance (simplified)
        this.updateFeatureImportance(tree);
      }
    }
    
    predict(point, nTreesToUse = this.trees.length) {
      const votes = {};
      const treesToUse = Math.min(nTreesToUse, this.trees.length);
      
      for (let i = 0; i < treesToUse; i++) {
        const prediction = this.trees[i].predict(point);
        votes[prediction] = (votes[prediction] || 0) + 1;
      }
      
      return Object.keys(votes).reduce((a, b) => votes[a] > votes[b] ? a : b);
    }
    
    updateFeatureImportance(tree) {
      // Simplified feature importance calculation
      const countFeatures = (node) => {
        if (!node) return;
        
        if (node.splitFeature) {
          this.featureImportance[node.splitFeature] += node.impurity * node.samples;
          countFeatures(node.left);
          countFeatures(node.right);
        }
      };
      
      countFeatures(tree);
    }
    
    getNormalizedFeatureImportance() {
      const total = this.featureImportance.x + this.featureImportance.y;
      return {
        x: this.featureImportance.x / total,
        y: this.featureImportance.y / total
      };
    }
  }
  
  // Enhanced decision boundary drawing for ensemble
  const drawEnsembleBoundary = (forest, progress = 1, nTreesToUse = null) => {
    if (!params.show_ensemble) return;
    
    ctx.save();
    
    // Draw decision regions as a wave effect
    const resolution = 40;
    const cellWidth = (width - 350 - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        const predictedClass = forest.predict({ x, y }, nTreesToUse);
        const color = COLORS.spectrum[predictedClass % COLORS.spectrum.length];
        
        // Wave effect based on distance from center
        const centerX = resolution / 2;
        const centerY = resolution / 2;
        const distance = Math.sqrt((i - centerX) ** 2 + (j - centerY) ** 2);
        const waveProgress = MathUtils.clamp(progress * 2 - distance / resolution, 0, 1);
        
        ctx.fillStyle = color + Math.floor(50 * waveProgress).toString(16).padStart(2, '0');
        ctx.fillRect(
          toCanvasX(x) - cellWidth / 2,
          toCanvasY(y) - cellHeight / 2,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Enhanced individual tree boundary drawing
  const drawTreeBoundary = (tree, progress = 1, treeIndex = 0) => {
    if (!params.show_individual_trees) return;
    
    ctx.save();
    ctx.globalAlpha = 0.3 * progress;
    
    // Draw decision regions for individual tree
    const resolution = 40;
    const cellWidth = (width - 350 - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        const predictedClass = tree.predict({ x, y });
        const color = COLORS.spectrum[predictedClass % COLORS.spectrum.length];
        
        ctx.fillStyle = color;
        ctx.fillRect(
          toCanvasX(x) - cellWidth / 2,
          toCanvasY(y) - cellHeight / 2,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Enhanced feature importance drawing
  const drawFeatureImportance = (forest, progress = 1) => {
    if (!params.show_importance) return;
    
    ctx.save();
    
    // Feature importance area
    const importanceX = width - 320;
    const importanceY = height - 150;
    const importanceWidth = 300;
    const importanceHeight = 100;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(importanceX, importanceY, importanceWidth, importanceHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(importanceX, importanceY, importanceWidth, importanceHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Feature Importance', importanceX + importanceWidth / 2, importanceY - 10);
    
    // Get normalized importance
    const importance = forest.getNormalizedFeatureImportance();
    
    // Draw bars
    const barWidth = 80;
    const barSpacing = 40;
    const maxBarHeight = 60;
    
    // Feature 1 (X)
    ctx.fillStyle = COLORS.primary;
    const barHeightX = importance.x * maxBarHeight * progress;
    ctx.fillRect(
      importanceX + importanceWidth / 2 - barSpacing - barWidth,
      importanceY + importanceHeight - barHeightX,
      barWidth,
      barHeightX
    );
    
    // Feature 2 (Y)
    ctx.fillStyle = COLORS.secondary;
    const barHeightY = importance.y * maxBarHeight * progress;
    ctx.fillRect(
      importanceX + importanceWidth / 2 + barSpacing,
      importanceY + importanceHeight - barHeightY,
      barWidth,
      barHeightY
    );
    
    // Draw labels
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Feature 1 (X)', importanceX + importanceWidth / 2 - barSpacing - barWidth / 2, importanceY + importanceHeight + 15);
    ctx.fillText('Feature 2 (Y)', importanceX + importanceWidth / 2 + barSpacing + barWidth / 2, importanceY + importanceHeight + 15);
    
    // Draw values
    ctx.fillText(importance.x.toFixed(3), importanceX + importanceWidth / 2 - barSpacing - barWidth / 2, importanceY + importanceHeight - barHeightX - 5);
    ctx.fillText(importance.y.toFixed(3), importanceX + importanceWidth / 2 + barSpacing + barWidth / 2, importanceY + importanceHeight - barHeightY - 5);
    
    ctx.restore();
  };
  
  // Enhanced OOB error drawing
  const drawOOBError = (forest, progress = 1) => {
    if (!params.show_oob || forest.oobErrors.length === 0) return;
    
    ctx.save();
    
    // OOB error area
    const oobX = width - 320;
    const oobY = 50;
    const oobWidth = 300;
    const oobHeight = 150;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(oobX, oobY, oobWidth, oobHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(oobX, oobY, oobWidth, oobHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Out-of-Bag Error', oobX + oobWidth / 2, oobY - 10);
    
    // Find min and max error for scaling
    const maxError = Math.max(...forest.oobErrors);
    const minError = Math.min(...forest.oobErrors);
    const errorRange = maxError - minError || 0.1; // Avoid division by zero
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(oobX + 30, oobY + 20);
    ctx.lineTo(oobX + 30, oobY + oobHeight - 20);
    ctx.lineTo(oobX + oobWidth - 20, oobY + oobHeight - 20);
    ctx.stroke();
    
    // Draw labels
    ctx.textAlign = 'right';
    ctx.fillText(minError.toFixed(2), oobX + 25, oobY + oobHeight - 20);
    ctx.fillText(maxError.toFixed(2), oobX + 25, oobY + 20);
    ctx.textAlign = 'center';
    ctx.fillText('Number of Trees', oobX + oobWidth / 2, oobY + oobHeight);
    
    // Draw OOB error curve
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const pointsToShow = Math.floor(forest.oobErrors.length * progress);
    const visibleErrors = forest.oobErrors.slice(0, pointsToShow);
    
    visibleErrors.forEach((error, i) => {
      const x = oobX + 30 + (oobWidth - 50) * (i / (forest.oobErrors.length - 1));
      const y = oobY + oobHeight - 20 - (oobHeight - 40) * ((error - minError) / errorRange);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw current point
    if (pointsToShow > 0) {
      const currentError = forest.oobErrors[pointsToShow - 1];
      const x = oobX + 30 + (oobWidth - 50) * ((pointsToShow - 1) / (forest.oobErrors.length - 1));
      const y = oobY + oobHeight - 20 - (oobHeight - 40) * ((currentError - minError) / errorRange);
      
      ctx.fillStyle = COLORS.accent;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw current error value
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(currentError.toFixed(4), x + 5, y);
    }
    
    ctx.restore();
  };
  
  // Animate the random forest building with enhanced cinematic effects
  const animateForestBuilding = () => {
    const forest = new RandomForest();
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing class by class
    const classes = [...new Set(data.map(p => p.label))];
    
    classes.forEach((classLabel, classIndex) => {
      timeline.add({
        duration: 800,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points from previous classes
          for (let i = 0; i < classIndex; i++) {
            const classPoints = data.filter(p => p.label === classes[i]);
            drawDataPoints(classPoints, 1);
          }
          
          // Draw points from current class with animation
          const classPoints = data.filter(p => p.label === classLabel);
          drawDataPoints(classPoints, progress);
        }
      }, { delay: classIndex * 300 });
    });
    
    // Phase 2: Animate trees being built one by one
    for (let treeIndex = 0; treeIndex < params.n_trees; treeIndex++) {
      timeline.add({
        duration: 1000,
        easing: 'easeInOutQuad',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          
          // Draw grid and points
          drawGrid();
          drawDataPoints(data, 1);
          
          // Train forest up to current tree
          if (progress > 0.5 && Math.floor(progress * params.n_trees) > forest.trees.length) {
            forest.train(data, Math.floor(progress * params.n_trees), params.max_depth, params.bootstrap_ratio);
          }
          
          // Draw ensemble boundary with trees built so far
          if (forest.trees.length > 0) {
            drawEnsembleBoundary(forest, progress, Math.floor(progress * params.n_trees));
          }
          
          // Draw individual tree if enabled
          if (params.show_individual_trees && forest.trees.length > treeIndex) {
            drawTreeBoundary(forest.trees[treeIndex], progress, treeIndex);
          }
          
          // Draw feature importance if enabled
          if (params.show_importance) {
            drawFeatureImportance(forest, progress);
          }
          
          // Draw OOB error if enabled
          if (params.show_oob) {
            drawOOBError(forest, progress);
          }
          
          // Draw tree count
          ctx.fillStyle = COLORS.text;
          ctx.font = '14px Arial';
          ctx.textAlign = 'left';
          ctx.fillText(`Trees: ${Math.floor(progress * params.n_trees)}/${params.n_trees}`, 60, 30);
        }
      }, { delay: treeIndex * 500 });
    }
    
    // Phase 3: Final reveal with all elements
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(data, 1);
        drawEnsembleBoundary(forest, 1);
        
        if (params.show_individual_trees) {
          // Show all individual trees with low opacity
          forest.trees.forEach((tree, i) => {
            drawTreeBoundary(tree, 0.2, i);
          });
        }
        
        if (params.show_importance) {
          drawFeatureImportance(forest, 1);
        }
        
        if (params.show_oob) {
          drawOOBError(forest, 1);
        }
        
        // Draw final stats
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Final Forest: ${params.n_trees} trees`, 60, 30);
        ctx.fillText(`Max Depth: ${params.max_depth}`, 60, 50);
        ctx.fillText(`Training Accuracy: ${calculateAccuracy(forest, data).toFixed(2)}%`, 60, 70);
        
        if (forest.oobErrors.length > 0) {
          ctx.fillText(`Final OOB Error: ${forest.oobErrors[forest.oobErrors.length - 1].toFixed(4)}`, 60, 90);
        }
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Helper function to calculate accuracy
  const calculateAccuracy = (forest, data) => {
    let correct = 0;
    data.forEach(point => {
      if (forest.predict(point) == point.label) {
        correct++;
      }
    });
    return (correct / data.length) * 100;
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'ensemble-view':
      params.show_individual_trees = false;
      params.show_importance = false;
      params.show_oob = false;
      animateForestBuilding();
      break;
      
    case 'individual-tree':
      params.show_individual_trees = true;
      params.show_ensemble = false;
      params.show_importance = false;
      params.show_oob = false;
      animateForestBuilding();
      break;
      
    case 'feature-importance':
      params.show_individual_trees = false;
      params.show_ensemble = false;
      params.show_importance = true;
      params.show_oob = false;
      animateForestBuilding();
      break;
      
    case 'oob-error':
      params.show_individual_trees = false;
      params.show_ensemble = false;
      params.show_importance = false;
      params.show_oob = true;
      animateForestBuilding();
      break;
      
    case 'all':
      params.show_individual_trees = true;
      params.show_ensemble = true;
      params.show_importance = true;
      params.show_oob = true;
      animateForestBuilding();
      break;
      
    default:
      animateForestBuilding();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Trees',
        type: 'range',
        min: 1,
        max: 50,
        step: 1,
        value: params.n_trees,
        onChange: (value) => {
          params.n_trees = parseInt(value);
          visualizers['random-forest'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Max Tree Depth',
        type: 'range',
        min: 1,
        max: 8,
        step: 1,
        value: params.max_depth,
        onChange: (value) => {
          params.max_depth = parseInt(value);
          visualizers['random-forest'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Bootstrap Ratio',
        type: 'range',
        min: 0.1,
        max: 1,
        step: 0.1,
        value: params.bootstrap_ratio,
        onChange: (value) => {
          params.bootstrap_ratio = parseFloat(value);
          visualizers['random-forest'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Individual Trees',
        type: 'checkbox',
        checked: params.show_individual_trees,
        onChange: (value) => {
          params.show_individual_trees = value;
          visualizers['random-forest'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Ensemble',
        type: 'checkbox',
        checked: params.show_ensemble,
        onChange: (value) => {
          params.show_ensemble = value;
          visualizers['random-forest'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Feature Importance',
        type: 'checkbox',
        checked: params.show_importance,
        onChange: (value) => {
          params.show_importance = value;
          visualizers['random-forest'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show OOB Error',
        type: 'checkbox',
        checked: params.show_oob,
        onChange: (value) => {
          params.show_oob = value;
          visualizers['random-forest'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'ensemble-view', label: 'Ensemble View', selected: visualizationType === 'ensemble-view' },
          { value: 'individual-tree', label: 'Individual Trees', selected: visualizationType === 'individual-tree' },
          { value: 'feature-importance', label: 'Feature Importance', selected: visualizationType === 'feature-importance' },
          { value: 'oob-error', label: 'OOB Error', selected: visualizationType === 'oob-error' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizers['random-forest'](containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'Random Forest Parameters',
      description: 'Adjust parameters to see how they affect the random forest model.'
    });
  }
};

// =============================================
// Enhanced K-Nearest Neighbors Visualizations
// =============================================
function visualizeKNN(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 100,
    n_classes: 3,
    k_value: 5,
    distance_metric: 'euclidean',
    weighted: false,
    show_boundary: true,
    show_voronoi: false,
    show_neighbors: true,
    highlight_neighbors: true,
    animation_duration: 1000,
    interactive: true,
    controlsContainer: null,
    distribution: 'concentric',
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
  const data = DataSimulator.generateDecisionTreeData({
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
  const padding = 0.5;
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
      ctx.fillText('Feature 1 (X)', width / 2, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing with class-by-class animation
  const drawDataPoints = (points, progress = 1, highlight = null) => {
    ctx.save();
    
    // Group points by class for sequential animation
    const pointsByClass = {};
    points.forEach(point => {
      if (!pointsByClass[point.label]) pointsByClass[point.label] = [];
      pointsByClass[point.label].push(point);
    });
    
    // Draw each class with different timing
    Object.keys(pointsByClass).forEach((classLabel, classIndex) => {
      const classPoints = pointsByClass[classLabel];
      const classProgress = MathUtils.clamp((progress - classIndex * 0.2) * 1.25, 0, 1);
      
      classPoints.forEach(point => {
        // Dim points that are not in the current highlight
        let alpha = classProgress;
        if (highlight && !highlight.includes(point)) {
          alpha *= 0.3;
        }
        
        ctx.globalAlpha = alpha;
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
    });
    
    ctx.restore();
  };
  
  // Enhanced distance calculation with different metrics
  const calculateDistance = (point1, point2, metric = 'euclidean') => {
    switch (metric) {
      case 'manhattan':
        return Math.abs(point1.x - point2.x) + Math.abs(point1.y - point2.y);
      case 'chebyshev':
        return Math.max(Math.abs(point1.x - point2.x), Math.abs(point1.y - point2.y));
      default: // euclidean
        return Math.sqrt(Math.pow(point1.x - point2.x, 2) + Math.pow(point1.y - point2.y, 2));
    }
  };
  
  // Enhanced KNN prediction
  const predictKNN = (point, data, k, metric = 'euclidean', weighted = false) => {
    // Calculate distances to all points
    const distances = data.map((p, i) => ({
      index: i,
      distance: calculateDistance(point, p, metric),
      label: p.label
    }));
    
    // Sort by distance
    distances.sort((a, b) => a.distance - b.distance);
    
    // Get k nearest neighbors
    const neighbors = distances.slice(0, k);
    
    // Count votes (weighted or unweighted)
    const votes = {};
    neighbors.forEach(neighbor => {
      const weight = weighted ? 1 / (neighbor.distance + 0.001) : 1; // Avoid division by zero
      votes[neighbor.label] = (votes[neighbor.label] || 0) + weight;
    });
    
    // Return majority class
    return Object.keys(votes).reduce((a, b) => votes[a] > votes[b] ? a : b);
  };
  
  // Enhanced decision boundary drawing
  const drawDecisionBoundary = (progress = 1) => {
    if (!params.show_boundary) return;
    
    ctx.save();
    ctx.globalAlpha = 0.6 * progress;
    
    // Draw decision regions
    const resolution = 40;
    const cellWidth = (width - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        const predictedClass = predictKNN(
          { x, y }, 
          data, 
          params.k_value, 
          params.distance_metric, 
          params.weighted
        );
        
        const color = COLORS.spectrum[predictedClass % COLORS.spectrum.length];
        
        ctx.fillStyle = color;
        ctx.fillRect(
          toCanvasX(x) - cellWidth / 2,
          toCanvasY(y) - cellHeight / 2,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Enhanced Voronoi diagram drawing
  const drawVoronoi = (progress = 1) => {
    if (!params.show_voronoi) return;
    
    ctx.save();
    ctx.globalAlpha = 0.2 * progress;
    
    // Simple Voronoi diagram based on nearest neighbor
    const resolution = 60;
    const cellWidth = (width - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        // Find nearest training point
        let minDistance = Infinity;
        let nearestPoint = null;
        
        data.forEach(point => {
          const distance = calculateDistance({ x, y }, point, params.distance_metric);
          if (distance < minDistance) {
            minDistance = distance;
            nearestPoint = point;
          }
        });
        
        if (nearestPoint) {
          const color = COLORS.spectrum[nearestPoint.label % COLORS.spectrum.length];
          ctx.fillStyle = color;
          ctx.fillRect(
            toCanvasX(x) - cellWidth / 2,
            toCanvasY(y) - cellHeight / 2,
            cellWidth,
            cellHeight
          );
        }
      }
    }
    
    ctx.restore();
  };
  
  // Enhanced neighbor highlighting
  const highlightNeighbors = (point, progress = 1) => {
    if (!params.show_neighbors || !params.highlight_neighbors) return;
    
    ctx.save();
    
    // Calculate distances to all points
    const distances = data.map(p => ({
      point: p,
      distance: calculateDistance(point, p, params.distance_metric)
    }));
    
    // Sort by distance
    distances.sort((a, b) => a.distance - b.distance);
    
    // Get k nearest neighbors
    const neighbors = distances.slice(0, params.k_value);
    
    // Draw lines to neighbors with animation
    neighbors.forEach((neighbor, i) => {
      const neighborProgress = MathUtils.clamp((progress - i * 0.1) * 1.5, 0, 1);
      
      if (neighborProgress > 0) {
        ctx.globalAlpha = 0.7 * neighborProgress;
        ctx.strokeStyle = COLORS.accent;
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        
        ctx.beginPath();
        ctx.moveTo(toCanvasX(point.x), toCanvasY(point.y));
        ctx.lineTo(toCanvasX(neighbor.point.x), toCanvasY(neighbor.point.y));
        ctx.stroke();
        
        ctx.setLineDash([]);
        
        // Draw distance circles
        if (params.distance_metric === 'euclidean') {
          ctx.globalAlpha = 0.2 * neighborProgress;
          ctx.strokeStyle = COLORS.accent;
          ctx.lineWidth = 1;
          
          ctx.beginPath();
          ctx.arc(
            toCanvasX(point.x), 
            toCanvasY(point.y), 
            toCanvasX(neighbor.distance) - toCanvasX(0), 
            0, 
            Math.PI * 2
          );
          ctx.stroke();
        }
      }
    });
    
    // Highlight neighbor points
    neighbors.forEach((neighbor, i) => {
      const neighborProgress = MathUtils.clamp((progress - i * 0.1) * 1.5, 0, 1);
      
      if (neighborProgress > 0) {
        ctx.globalAlpha = neighborProgress;
        ctx.beginPath();
        ctx.arc(toCanvasX(neighbor.point.x), toCanvasY(neighbor.point.y), 8, 0, Math.PI * 2);
        
        // Pulse effect for highlighted points
        const pulse = 1 + 0.2 * Math.sin(Date.now() / 200);
        ctx.fillStyle = COLORS.accent;
        ctx.fill();
        
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    });
    
    ctx.restore();
  };
  
  // Enhanced distance metric visualization
  const drawDistanceMetric = (point, progress = 1) => {
    ctx.save();
    
    // Draw different distance metrics visualization
    const metrics = ['euclidean', 'manhattan', 'chebyshev'];
    const metricY = 100;
    
    metrics.forEach((metric, i) => {
      const metricX = 50 + i * 250;
      const isCurrent = metric === params.distance_metric;
      
      // Draw title
      ctx.fillStyle = isCurrent ? COLORS.primary : COLORS.text;
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(metric.charAt(0).toUpperCase() + metric.slice(1), metricX + 100, metricY - 30);
      
      // Draw distance visualization
      ctx.globalAlpha = isCurrent ? 0.8 : 0.3;
      
      if (metric === 'euclidean') {
        // Euclidean circles
        ctx.strokeStyle = COLORS.primary;
        ctx.lineWidth = isCurrent ? 2 : 1;
        
        for (let r = 20; r <= 80; r += 20) {
          ctx.beginPath();
          ctx.arc(metricX + 100, metricY, r, 0, Math.PI * 2);
          ctx.stroke();
        }
      } else if (metric === 'manhattan') {
        // Manhattan diamonds
        ctx.strokeStyle = COLORS.secondary;
        ctx.lineWidth = isCurrent ? 2 : 1;
        
        for (let r = 20; r <= 80; r += 20) {
          ctx.beginPath();
          ctx.moveTo(metricX + 100, metricY - r);
          ctx.lineTo(metricX + 100 + r, metricY);
          ctx.lineTo(metricX + 100, metricY + r);
          ctx.lineTo(metricX + 100 - r, metricY);
          ctx.closePath();
          ctx.stroke();
        }
      } else if (metric === 'chebyshev') {
        // Chebyshev squares
        ctx.strokeStyle = COLORS.accent;
        ctx.lineWidth = isCurrent ? 2 : 1;
        
        for (let r = 20; r <= 80; r += 20) {
          ctx.beginPath();
          ctx.rect(metricX + 100 - r, metricY - r, r * 2, r * 2);
          ctx.stroke();
        }
      }
    });
    
    ctx.restore();
  };
  
  // Enhanced k-value comparison
  const drawKComparison = (progress = 1) => {
    ctx.save();
    
    // Draw decision boundaries for different k values
    const kValues = [1, 3, 5, 7, 9, 11];
    const comparisonWidth = width / 3;
    const comparisonHeight = height / 2;
    
    kValues.forEach((k, i) => {
      const row = Math.floor(i / 3);
      const col = i % 3;
      const compX = col * comparisonWidth;
      const compY = row * comparisonHeight;
      
      // Draw title
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`k = ${k}`, compX + comparisonWidth / 2, compY + 20);
      
      // Draw decision boundary
      ctx.globalAlpha = 0.6;
      
      const resolution = 20;
      const cellWidth = comparisonWidth / resolution;
      const cellHeight = comparisonHeight / resolution;
      
      for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
          const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
          const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
          
          const predictedClass = predictKNN(
            { x, y }, 
            data, 
            k, 
            params.distance_metric, 
            params.weighted
          );
          
          const color = COLORS.spectrum[predictedClass % COLORS.spectrum.length];
          
          ctx.fillStyle = color;
          ctx.fillRect(
            compX + i * cellWidth,
            compY + 30 + j * cellHeight,
            cellWidth,
            cellHeight
          );
        }
      }
      
      // Draw data points
      ctx.globalAlpha = 1;
      data.forEach(point => {
        ctx.beginPath();
        ctx.arc(
          compX + MathUtils.map(point.x, bounds.xMin, bounds.xMax, 0, comparisonWidth),
          compY + 30 + MathUtils.map(point.y, bounds.yMin, bounds.yMax, 0, comparisonHeight),
          2, 0, Math.PI * 2
        );
        
        const color = COLORS.spectrum[point.label % COLORS.spectrum.length];
        ctx.fillStyle = color;
        ctx.fill();
        
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 0.5;
        ctx.stroke();
      });
    });
    
    ctx.restore();
  };
  
  // Animate the KNN visualization with enhanced cinematic effects
  const animateKNN = () => {
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing class by class
    const classes = [...new Set(data.map(p => p.label))];
    
    classes.forEach((classLabel, classIndex) => {
      timeline.add({
        duration: 800,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points from previous classes
          for (let i = 0; i < classIndex; i++) {
            const classPoints = data.filter(p => p.label === classes[i]);
            drawDataPoints(classPoints, 1);
          }
          
          // Draw points from current class with animation
          const classPoints = data.filter(p => p.label === classLabel);
          drawDataPoints(classPoints, progress);
        }
      }, { delay: classIndex * 300 });
    });
    
    // Phase 2: Animate decision boundary
    timeline.add({
      duration: 2000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid and points
        drawGrid();
        drawDataPoints(data, 1);
        
        // Draw decision boundary
        drawDecisionBoundary(progress);
        
        // Draw Voronoi if enabled
        if (params.show_voronoi) {
          drawVoronoi(progress);
        }
        
        // Draw distance metric visualization
        drawDistanceMetric({ x: 0, y: 0 }, progress);
        
        // Draw k value
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`k = ${params.k_value}`, 60, 30);
        ctx.fillText(`Distance: ${params.distance_metric}`, 60, 50);
        ctx.fillText(`Weighted: ${params.weighted ? 'Yes' : 'No'}`, 60, 70);
      }
    });
    
    // Phase 3: Interactive query point
    let queryPoint = null;
    
    timeline.add({
      duration: 1000,
      onStart: () => {
        // Add random query point
        queryPoint = {
          x: MathUtils.random(bounds.xMin + 0.2, bounds.xMax - 0.2),
          y: MathUtils.random(bounds.yMin + 0.2, bounds.yMax - 0.2),
          label: null
        };
        
        // Predict class for query point
        queryPoint.label = predictKNN(
          queryPoint, 
          data, 
          params.k_value, 
          params.distance_metric, 
          params.weighted
        );
      },
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid, points, and boundary
        drawGrid();
        drawDataPoints(data, 1);
        drawDecisionBoundary(1);
        
        if (params.show_voronoi) {
          drawVoronoi(1);
        }
        
        // Draw query point with animation
        ctx.globalAlpha = progress;
        ctx.beginPath();
        ctx.arc(toCanvasX(queryPoint.x), toCanvasY(queryPoint.y), 10, 0, Math.PI * 2);
        
        // Pulse effect for query point
        const pulse = 1 + 0.2 * Math.sin(Date.now() / 200);
        ctx.fillStyle = COLORS.spectrum[queryPoint.label % COLORS.spectrum.length];
        ctx.fill();
        
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Highlight neighbors
        highlightNeighbors(queryPoint, progress);
        
        // Draw prediction
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Predicted class: ${queryPoint.label}`, 60, 30);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'decision-boundary':
      params.show_voronoi = false;
      animateKNN();
      break;
      
    case 'voronoi':
      params.show_boundary = false;
      params.show_voronoi = true;
      animateKNN();
      break;
      
    case 'distance-metrics':
      // Special visualization for distance metrics comparison
      const animateDistanceMetrics = () => {
        const timeline = AnimationSystem.createTimeline();
        
        // Animate through different distance metrics
        const metrics = ['euclidean', 'manhattan', 'chebyshev'];
        
        metrics.forEach((metric, i) => {
          timeline.add({
            duration: 1500,
            onStart: () => {
              params.distance_metric = metric;
            },
            onUpdate: (progress) => {
              ctx.clearRect(0, 0, width, height);
              
              // Draw grid and points
              drawGrid();
              drawDataPoints(data, 1);
              
              // Draw decision boundary
              drawDecisionBoundary(1);
              
              // Draw distance metric visualization
              drawDistanceMetric({ x: 0, y: 0 }, 1);
              
              // Draw current metric
              ctx.fillStyle = COLORS.text;
              ctx.font = '16px Arial';
              ctx.textAlign = 'left';
              ctx.fillText(`Distance Metric: ${metric}`, 60, 30);
            }
          }, { delay: i * 2000 });
        });
        
        timeline.play();
      };
      
      animateDistanceMetrics();
      break;
      
    case 'k-comparison':
      // Special visualization for k value comparison
      const animateKComparison = () => {
        const timeline = AnimationSystem.createTimeline();
        
        // Animate through different k values
        const kValues = [1, 3, 5, 7, 9, 11, 15];
        
        kValues.forEach((k, i) => {
          timeline.add({
            duration: 1500,
            onStart: () => {
              params.k_value = k;
            },
            onUpdate: (progress) => {
              ctx.clearRect(0, 0, width, height);
              
              // Draw grid and points
              drawGrid();
              drawDataPoints(data, 1);
              
              // Draw decision boundary
              drawDecisionBoundary(1);
              
              // Draw k value
              ctx.fillStyle = COLORS.text;
              ctx.font = '16px Arial';
              ctx.textAlign = 'left';
              ctx.fillText(`k = ${k}`, 60, 30);
            }
          }, { delay: i * 2000 });
        });
        
        timeline.play();
      };
      
      animateKComparison();
      break;
      
    case 'all':
      params.show_boundary = true;
      params.show_voronoi = true;
      animateKNN();
      break;
      
    default:
      animateKNN();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'k Value',
        type: 'range',
        min: 1,
        max: 15,
        step: 2,
        value: params.k_value,
        onChange: (value) => {
          params.k_value = parseInt(value);
          visualizers['knn'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Distance Metric',
        type: 'select',
        options: [
          { value: 'euclidean', label: 'Euclidean', selected: params.distance_metric === 'euclidean' },
          { value: 'manhattan', label: 'Manhattan', selected: params.distance_metric === 'manhattan' },
          { value: 'chebyshev', label: 'Chebyshev', selected: params.distance_metric === 'chebyshev' }
        ],
        onChange: (value) => {
          params.distance_metric = value;
          visualizers['knn'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Weighted Voting',
        type: 'checkbox',
        checked: params.weighted,
        onChange: (value) => {
          params.weighted = value;
          visualizers['knn'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision Boundary',
        type: 'checkbox',
        checked: params.show_boundary,
        onChange: (value) => {
          params.show_boundary = value;
          visualizers['knn'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Voronoi Diagram',
        type: 'checkbox',
        checked: params.show_voronoi,
        onChange: (value) => {
          params.show_voronoi = value;
          visualizers['knn'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Highlight Neighbors',
        type: 'checkbox',
        checked: params.highlight_neighbors,
        onChange: (value) => {
          params.highlight_neighbors = value;
          visualizers['knn'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'decision-boundary', label: 'Decision Boundary', selected: visualizationType === 'decision-boundary' },
          { value: 'voronoi', label: 'Voronoi Diagram', selected: visualizationType === 'voronoi' },
          { value: 'distance-metrics', label: 'Distance Comparison', selected: visualizationType === 'distance-metrics' },
          { value: 'k-comparison', label: 'K Value Comparison', selected: visualizationType === 'k-comparison' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizers['knn'](containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'KNN Parameters',
      description: 'Adjust parameters to see how they affect the K-Nearest Neighbors model.'
    });
  }
};

// =============================================
// Enhanced Support Vector Machines Visualizations
// =============================================
function visualizeSVM(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 100,
    n_classes: 2,
    kernel: 'rbf',
    C: 1.0,
    gamma: 'scale',
    degree: 3,
    show_boundary: true,
    show_margin: true,
    show_support_vectors: true,
    show_contours: false,
    animation_duration: 2000,
    interactive: true,
    controlsContainer: null,
    distribution: 'concentric',
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
  const data = DataSimulator.generateDecisionTreeData({
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
  const padding = 0.5;
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
      ctx.fillText('Feature 1 (X)', width / 2, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing with class-by-class animation
  const drawDataPoints = (points, progress = 1, highlight = null) => {
    ctx.save();
    
    // Group points by class for sequential animation
    const pointsByClass = {};
    points.forEach(point => {
      if (!pointsByClass[point.label]) pointsByClass[point.label] = [];
      pointsByClass[point.label].push(point);
    });
    
    // Draw each class with different timing
    Object.keys(pointsByClass).forEach((classLabel, classIndex) => {
      const classPoints = pointsByClass[classLabel];
      const classProgress = MathUtils.clamp((progress - classIndex * 0.2) * 1.25, 0, 1);
      
      classPoints.forEach(point => {
        // Dim points that are not in the current highlight
        let alpha = classProgress;
        if (highlight && !highlight.includes(point)) {
          alpha *= 0.3;
        }
        
        ctx.globalAlpha = alpha;
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
    });
    
    ctx.restore();
  };
  
  // Enhanced kernel functions
  const kernelFunction = (x1, x2, kernelType = 'linear', gamma = 1, degree = 3) => {
    switch (kernelType) {
      case 'poly':
        return Math.pow(gamma * (x1.x * x2.x + x1.y * x2.y) + 1, degree);
      case 'rbf':
        const distance = Math.pow(x1.x - x2.x, 2) + Math.pow(x1.y - x2.y, 2);
        return Math.exp(-gamma * distance);
      default: // linear
        return x1.x * x2.x + x1.y * x2.y;
    }
  };
  
  // Simplified SVM training (for visualization purposes)
  const trainSVM = (data, kernelType = 'linear', C = 1.0, gamma = 1, degree = 3) => {
    // This is a simplified version for visualization
    // In a real implementation, you would use a proper optimization algorithm
    
    // Find support vectors (points near the decision boundary)
    const supportVectors = [];
    const margin = 0.2; // Simplified margin
    
    data.forEach(point => {
      // Simplified: points near the origin are support vectors
      const distance = Math.sqrt(point.x * point.x + point.y * point.y);
      if (Math.abs(distance - 1) < margin) {
        supportVectors.push(point);
      }
    });
    
    return {
      supportVectors,
      kernelType,
      C,
      gamma,
      degree,
      predict: (point) => {
        // Simplified prediction based on kernel and support vectors
        let score = 0;
        
        supportVectors.forEach(sv => {
          const kernelValue = kernelFunction(point, sv, kernelType, gamma, degree);
          score += sv.label * kernelValue; // Simplified: assume binary classification with labels -1 and 1
        });
        
        return score >= 0 ? 1 : 0;
      }
    };
  };
  
  // Enhanced decision boundary drawing
  const drawDecisionBoundary = (svm, progress = 1) => {
    if (!params.show_boundary) return;
    
    ctx.save();
    ctx.globalAlpha = 0.6 * progress;
    
    // Draw decision regions
    const resolution = 40;
    const cellWidth = (width - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        const predictedClass = svm.predict({ x, y });
        const color = COLORS.spectrum[predictedClass % COLORS.spectrum.length];
        
        ctx.fillStyle = color;
        ctx.fillRect(
          toCanvasX(x) - cellWidth / 2,
          toCanvasY(y) - cellHeight / 2,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Enhanced margin drawing
  const drawMargin = (svm, progress = 1) => {
    if (!params.show_margin) return;
    
    ctx.save();
    
    // Draw margin lines (simplified)
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    
    // Draw positive margin
    ctx.beginPath();
    ctx.arc(toCanvasX(0), toCanvasY(0), toCanvasX(1.2) - toCanvasX(0), 0, Math.PI * 2);
    ctx.stroke();
    
    // Draw negative margin
    ctx.beginPath();
    ctx.arc(toCanvasX(0), toCanvasY(0), toCanvasX(0.8) - toCanvasX(0), 0, Math.PI * 2);
    ctx.stroke();
    
    ctx.setLineDash([]);
    
    // Draw decision boundary (hyperplane)
    ctx.strokeStyle = COLORS.primary;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(toCanvasX(0), toCanvasY(0), toCanvasX(1) - toCanvasX(0), 0, Math.PI * 2);
    ctx.stroke();
    
    ctx.restore();
  };
  
  // Enhanced support vectors highlighting
  const drawSupportVectors = (svm, progress = 1) => {
    if (!params.show_support_vectors) return;
    
    ctx.save();
    
    // Draw support vectors with pulse effect
    svm.supportVectors.forEach((sv, i) => {
      const svProgress = MathUtils.clamp((progress - i * 0.1) * 1.5, 0, 1);
      
      if (svProgress > 0) {
        ctx.globalAlpha = svProgress;
        ctx.beginPath();
        ctx.arc(toCanvasX(sv.x), toCanvasY(sv.y), 10, 0, Math.PI * 2);
        
        // Pulse effect for support vectors
        const pulse = 1 + 0.2 * Math.sin(Date.now() / 200);
        ctx.fillStyle = COLORS.accent;
        ctx.fill();
        
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });
    
    ctx.restore();
  };
  
  // Enhanced kernel visualization
  const drawKernelVisualization = (svm, progress = 1) => {
    ctx.save();
    
    // Draw kernel visualization
    const kernelX = 50;
    const kernelY = 100;
    const kernelWidth = 200;
    const kernelHeight = 150;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(kernelX, kernelY, kernelWidth, kernelHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(kernelX, kernelY, kernelWidth, kernelHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`Kernel: ${svm.kernelType}`, kernelX + kernelWidth / 2, kernelY - 10);
    
    // Draw kernel visualization based on type
    ctx.globalAlpha = 0.8;
    
    if (svm.kernelType === 'linear') {
      // Linear kernel visualization
      ctx.strokeStyle = COLORS.primary;
      ctx.lineWidth = 2;
      
      ctx.beginPath();
      ctx.moveTo(kernelX + 20, kernelY + kernelHeight / 2);
      ctx.lineTo(kernelX + kernelWidth - 20, kernelY + kernelHeight / 2);
      ctx.stroke();
      
    } else if (svm.kernelType === 'poly') {
      // Polynomial kernel visualization
      ctx.strokeStyle = COLORS.secondary;
      ctx.lineWidth = 2;
      
      ctx.beginPath();
      for (let x = 0; x <= kernelWidth - 40; x++) {
        const xVal = x / (kernelWidth - 40);
        const yVal = Math.pow(xVal, svm.degree);
        const y = kernelY + kernelHeight - 20 - yVal * (kernelHeight - 40);
        
        if (x === 0) {
          ctx.moveTo(kernelX + 20 + x, y);
        } else {
          ctx.lineTo(kernelX + 20 + x, y);
        }
      }
      ctx.stroke();
      
    } else if (svm.kernelType === 'rbf') {
      // RBF kernel visualization
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2;
      
      ctx.beginPath();
      for (let x = 0; x <= kernelWidth - 40; x++) {
        const xVal = (x - (kernelWidth - 40) / 2) / ((kernelWidth - 40) / 2);
        const yVal = Math.exp(-svm.gamma * xVal * xVal);
        const y = kernelY + kernelHeight - 20 - yVal * (kernelHeight - 40);
        
        if (x === 0) {
          ctx.moveTo(kernelX + 20 + x, y);
        } else {
          ctx.lineTo(kernelX + 20 + x, y);
        }
      }
      ctx.stroke();
    }
    
    // Draw kernel parameters
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    
    if (svm.kernelType === 'poly') {
      ctx.fillText(`Degree: ${svm.degree}`, kernelX + 10, kernelY + kernelHeight + 15);
    } else if (svm.kernelType === 'rbf') {
      ctx.fillText(`Gamma: ${svm.gamma}`, kernelX + 10, kernelY + kernelHeight + 15);
    }
    
    ctx.restore();
  };
  
  // Enhanced parameter effects visualization
  const drawParameterEffects = (svm, progress = 1) => {
    ctx.save();
    
    // Draw parameter effects visualization
    const paramX = width - 250;
    const paramY = 100;
    const paramWidth = 200;
    const paramHeight = 150;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(paramX, paramY, paramWidth, paramHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(paramX, paramY, paramWidth, paramHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Parameter Effects', paramX + paramWidth / 2, paramY - 10);
    
    // Draw C parameter effect
    ctx.fillStyle = COLORS.primary;
    ctx.font = '12px Arial';
    ctx.textAlign = 'left';
    ctx.fillText(`C: ${svm.C}`, paramX + 10, paramY + 20);
    
    // Draw gamma parameter effect (if applicable)
    if (svm.kernelType === 'rbf') {
      ctx.fillStyle = COLORS.secondary;
      ctx.fillText(`Gamma: ${svm.gamma}`, paramX + 10, paramY + 40);
    }
    
    // Draw visual representation of parameter effects
    ctx.globalAlpha = 0.6;
    
    // C parameter (regularization)
    const cEffect = MathUtils.map(svm.C, 0.1, 10, 0.5, 0.9);
    ctx.fillStyle = COLORS.primary;
    ctx.beginPath();
    ctx.arc(
      paramX + paramWidth / 2,
      paramY + paramHeight / 2,
      paramWidth / 4 * cEffect,
      0, Math.PI * 2
    );
    ctx.fill();
    
    // Gamma parameter (for RBF)
    if (svm.kernelType === 'rbf') {
      const gammaEffect = MathUtils.map(svm.gamma, 0.1, 10, 0.1, 0.5);
      ctx.fillStyle = COLORS.secondary;
      ctx.beginPath();
      ctx.arc(
        paramX + paramWidth / 2,
        paramY + paramHeight / 2,
        paramWidth / 4 * gammaEffect,
        0, Math.PI * 2
      );
      ctx.fill();
    }
    
    ctx.restore();
  };
  
  // Animate the SVM visualization with enhanced cinematic effects
  const animateSVM = () => {
    // Train SVM
    const svm = trainSVM(
      data, 
      params.kernel, 
      params.C, 
      params.gamma === 'scale' ? 1 / (data.length * MathUtils.variance(data.map(p => p.x))) : params.gamma,
      params.degree
    );
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing class by class
    const classes = [...new Set(data.map(p => p.label))];
    
    classes.forEach((classLabel, classIndex) => {
      timeline.add({
        duration: 800,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points from previous classes
          for (let i = 0; i < classIndex; i++) {
            const classPoints = data.filter(p => p.label === classes[i]);
            drawDataPoints(classPoints, 1);
          }
          
          // Draw points from current class with animation
          const classPoints = data.filter(p => p.label === classLabel);
          drawDataPoints(classPoints, progress);
        }
      }, { delay: classIndex * 300 });
    });
    
    // Phase 2: Animate decision boundary
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid and points
        drawGrid();
        drawDataPoints(data, 1);
        
        // Draw decision boundary
        drawDecisionBoundary(svm, progress);
        
        // Draw margin if enabled
        if (params.show_margin) {
          drawMargin(svm, progress);
        }
        
        // Draw support vectors if enabled
        if (params.show_support_vectors) {
          drawSupportVectors(svm, progress);
        }
        
        // Draw kernel visualization
        drawKernelVisualization(svm, progress);
        
        // Draw parameter effects
        drawParameterEffects(svm, progress);
        
        // Draw SVM parameters
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Kernel: ${svm.kernelType}`, 60, 30);
        ctx.fillText(`C: ${svm.C}`, 60, 50);
        
        if (svm.kernelType === 'rbf') {
          ctx.fillText(`Gamma: ${svm.gamma}`, 60, 70);
        } else if (svm.kernelType === 'poly') {
          ctx.fillText(`Degree: ${svm.degree}`, 60, 70);
        }
      }
    });
    
    // Phase 3: Final reveal with all elements
    timeline.add({
      duration: 1000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(data, 1);
        drawDecisionBoundary(svm, 1);
        
        if (params.show_margin) {
          drawMargin(svm, 1);
        }
        
        if (params.show_support_vectors) {
          drawSupportVectors(svm, 1);
        }
        
        drawKernelVisualization(svm, 1);
        drawParameterEffects(svm, 1);
        
        // Draw final stats
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`SVM with ${svm.kernelType} kernel`, 60, 30);
        ctx.fillText(`Support Vectors: ${svm.supportVectors.length}`, 60, 50);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'decision-boundary':
      params.show_margin = false;
      params.show_support_vectors = false;
      animateSVM();
      break;
      
    case 'margin':
      params.show_boundary = false;
      params.show_support_vectors = false;
      animateSVM();
      break;
      
    case 'kernel-comparison':
      // Special visualization for kernel comparison
      const animateKernelComparison = () => {
        const timeline = AnimationSystem.createTimeline();
        
        // Animate through different kernels
        const kernels = ['linear', 'poly', 'rbf'];
        
        kernels.forEach((kernel, i) => {
          timeline.add({
            duration: 1500,
            onStart: () => {
              params.kernel = kernel;
            },
            onUpdate: (progress) => {
              ctx.clearRect(0, 0, width, height);
              
              // Draw grid and points
              drawGrid();
              drawDataPoints(data, 1);
              
              // Train and draw SVM with current kernel
              const svm = trainSVM(
                data, 
                kernel, 
                params.C, 
                params.gamma === 'scale' ? 1 / (data.length * MathUtils.variance(data.map(p => p.x))) : params.gamma,
                params.degree
              );
              
              drawDecisionBoundary(svm, 1);
              drawKernelVisualization(svm, 1);
              
              // Draw current kernel
              ctx.fillStyle = COLORS.text;
              ctx.font = '16px Arial';
              ctx.textAlign = 'left';
              ctx.fillText(`Kernel: ${kernel}`, 60, 30);
            }
          }, { delay: i * 2000 });
        });
        
        timeline.play();
      };
      
      animateKernelComparison();
      break;
      
    case 'parameter-effects':
      // Special visualization for parameter effects
      const animateParameterEffects = () => {
        const timeline = AnimationSystem.createTimeline();
        
        // Animate through different C values
        const cValues = [0.1, 1, 10];
        
        cValues.forEach((c, i) => {
          timeline.add({
            duration: 1500,
            onStart: () => {
              params.C = c;
            },
            onUpdate: (progress) => {
              ctx.clearRect(0, 0, width, height);
              
              // Draw grid and points
              drawGrid();
              drawDataPoints(data, 1);
              
              // Train and draw SVM with current C
              const svm = trainSVM(
                data, 
                params.kernel, 
                c, 
                params.gamma === 'scale' ? 1 / (data.length * MathUtils.variance(data.map(p => p.x))) : params.gamma,
                params.degree
              );
              
              drawDecisionBoundary(svm, 1);
              drawParameterEffects(svm, 1);
              
              // Draw current C value
              ctx.fillStyle = COLORS.text;
              ctx.font = '16px Arial';
              ctx.textAlign = 'left';
              ctx.fillText(`C: ${c}`, 60, 30);
            }
          }, { delay: i * 2000 });
        });
        
        // For RBF kernel, also animate gamma values
        if (params.kernel === 'rbf') {
          const gammaValues = [0.1, 1, 10];
          
          gammaValues.forEach((gamma, i) => {
            timeline.add({
              duration: 1500,
              onStart: () => {
                params.gamma = gamma;
              },
              onUpdate: (progress) => {
                ctx.clearRect(0, 0, width, height);
                
                // Draw grid and points
                drawGrid();
                drawDataPoints(data, 1);
                
                // Train and draw SVM with current gamma
                const svm = trainSVM(
                  data, 
                  params.kernel, 
                  params.C, 
                  gamma,
                  params.degree
                );
                
                drawDecisionBoundary(svm, 1);
                drawParameterEffects(svm, 1);
                
                // Draw current gamma value
                ctx.fillStyle = COLORS.text;
                ctx.font = '16px Arial';
                ctx.textAlign = 'left';
                ctx.fillText(`Gamma: ${gamma}`, 60, 30);
              }
            }, { delay: (i + cValues.length) * 2000 });
          });
        }
        
        timeline.play();
      };
      
      animateParameterEffects();
      break;
      
    case 'all':
      params.show_boundary = true;
      params.show_margin = true;
      params.show_support_vectors = true;
      animateSVM();
      break;
      
    default:
      animateSVM();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Kernel Type',
        type: 'select',
        options: [
          { value: 'linear', label: 'Linear', selected: params.kernel === 'linear' },
          { value: 'poly', label: 'Polynomial', selected: params.kernel === 'poly' },
          { value: 'rbf', label: 'RBF', selected: params.kernel === 'rbf' }
        ],
        onChange: (value) => {
          params.kernel = value;
          visualizers['svm'](containerId, visualizationType, params);
        }
      },
      {
        label: 'C Parameter',
        type: 'range',
        min: 0.1,
        max: 10,
        step: 0.1,
        value: params.C,
        onChange: (value) => {
          params.C = parseFloat(value);
          visualizers['svm'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Gamma',
        type: 'range',
        min: 0.1,
        max: 10,
        step: 0.1,
        value: params.gamma === 'scale' ? 1 : params.gamma,
        onChange: (value) => {
          params.gamma = parseFloat(value);
          visualizers['svm'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Degree (for Polynomial)',
        type: 'range',
        min: 2,
        max: 5,
        step: 1,
        value: params.degree,
        onChange: (value) => {
          params.degree = parseInt(value);
          visualizers['svm'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision Boundary',
        type: 'checkbox',
        checked: params.show_boundary,
        onChange: (value) => {
          params.show_boundary = value;
          visualizers['svm'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Margin',
        type: 'checkbox',
        checked: params.show_margin,
        onChange: (value) => {
          params.show_margin = value;
          visualizers['svm'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Support Vectors',
        type: 'checkbox',
        checked: params.show_support_vectors,
        onChange: (value) => {
          params.show_support_vectors = value;
          visualizers['svm'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'decision-boundary', label: 'Decision Boundary', selected: visualizationType === 'decision-boundary' },
          { value: 'margin', label: 'Margin Visualization', selected: visualizationType === 'margin' },
          { value: 'kernel-comparison', label: 'Kernel Comparison', selected: visualizationType === 'kernel-comparison' },
          { value: 'parameter-effects', label: 'Parameter Effects', selected: visualizationType === 'parameter-effects' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizers['svm'](containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'SVM Parameters',
      description: 'Adjust parameters to see how they affect the Support Vector Machine model.'
    });
  }
};

// =============================================$
// Quadratic Discriminant Analysis Visualization
// =============================================
function visualizeQDA(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 200,
    n_classes: 2,
    covariance_scale: 1.5,
    class_separation: 1.0,
    show_boundary: true,
    show_probability: false,
    show_comparison: false,
    animation_duration: 3000,
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
  
  // Generate QDA data with Gaussian distributions
  const generateQdaData = () => {
    const data = [];
    const centers = [];
    const covariances = [];
    
    // Create class centers with separation
    for (let i = 0; i < params.n_classes; i++) {
      const angle = (i / params.n_classes) * Math.PI * 2;
      const distance = params.class_separation * 2;
      centers.push({
        x: Math.cos(angle) * distance,
        y: Math.sin(angle) * distance
      });
      
      // Create different covariance matrices for each class
      const rotation = Math.random() * Math.PI;
      const scaleX = 1 + Math.random() * params.covariance_scale;
      const scaleY = 1 + Math.random() * params.covariance_scale;
      
      covariances.push({
        rotation,
        scaleX,
        scaleY
      });
    }
    
    // Generate samples for each class
    const samplesPerClass = Math.floor(params.n_samples / params.n_classes);
    
    for (let i = 0; i < params.n_classes; i++) {
      const center = centers[i];
      const covariance = covariances[i];
      
      for (let j = 0; j < samplesPerClass; j++) {
        // Generate point from multivariate normal distribution
        const x = MathUtils.gaussianRandom(0, 1);
        const y = MathUtils.gaussianRandom(0, 1);
        
        // Apply covariance transformation
        const rotatedX = x * Math.cos(covariance.rotation) - y * Math.sin(covariance.rotation);
        const rotatedY = x * Math.sin(covariance.rotation) + y * Math.cos(covariance.rotation);
        
        const scaledX = rotatedX * covariance.scaleX;
        const scaledY = rotatedY * covariance.scaleY;
        
        // Translate to class center
        const finalX = center.x + scaledX;
        const finalY = center.y + scaledY;
        
        data.push({ x: finalX, y: finalY, label: i });
      }
    }
    
    return { data, centers, covariances };
  };
  
  const { data, centers, covariances } = generateQdaData();
  
  // Find data bounds for scaling
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const padding = 2;
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
  
  // Enhanced data points drawing with class-by-class animation
  const drawDataPoints = (points, progress = 1) => {
    ctx.save();
    
    // Group points by class for sequential animation
    const pointsByClass = {};
    points.forEach(point => {
      if (!pointsByClass[point.label]) pointsByClass[point.label] = [];
      pointsByClass[point.label].push(point);
    });
    
    // Draw each class with different timing
    Object.keys(pointsByClass).forEach((classLabel, classIndex) => {
      const classPoints = pointsByClass[classLabel];
      const classProgress = MathUtils.clamp((progress - classIndex * 0.2) * 1.25, 0, 1);
      
      classPoints.forEach(point => {
        ctx.globalAlpha = classProgress;
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
    });
    
    ctx.restore();
  };
  
  // Draw Gaussian ellipses for each class
  const drawGaussianEllipses = (progress = 1) => {
    ctx.save();
    
    centers.forEach((center, i) => {
      const covariance = covariances[i];
      const color = COLORS.spectrum[i % COLORS.spectrum.length];
      
      // Draw multiple ellipses for contour effect
      for (let j = 1; j <= 3; j++) {
        const scale = j * 0.8;
        const ellipseProgress = MathUtils.clamp((progress - j * 0.2) * 2, 0, 1);
        
        if (ellipseProgress <= 0) continue;
        
        ctx.globalAlpha = 0.2 * ellipseProgress;
        ctx.fillStyle = color;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        
        ctx.beginPath();
        ctx.ellipse(
          toCanvasX(center.x),
          toCanvasY(center.y),
          covariance.scaleX * scale * 10,
          covariance.scaleY * scale * 10,
          covariance.rotation,
          0,
          Math.PI * 2
        );
        
        if (j === 3) {
          ctx.stroke(); // Outline for the outermost ellipse
        } else {
          ctx.fill(); // Fill for inner ellipses
        }
      }
      
      // Draw center point
      ctx.globalAlpha = progress;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(toCanvasX(center.x), toCanvasY(center.y), 6, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1;
      ctx.stroke();
    });
    
    ctx.restore();
  };
  
  // Draw quadratic decision boundary
  const drawDecisionBoundary = (progress = 1) => {
    if (!params.show_boundary) return;
    
    ctx.save();
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.globalAlpha = progress;
    
    // For QDA, we'll approximate the quadratic boundary by sampling points
    // where the difference between class probabilities is minimal
    
    // Create a grid of points
    const resolution = 100;
    const grid = [];
    
    for (let i = 0; i <= resolution; i++) {
      const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
      const row = [];
      
      for (let j = 0; j <= resolution; j++) {
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        // Calculate Mahalanobis distance to each class center
        const distances = centers.map((center, idx) => {
          const covariance = covariances[idx];
          
          // Rotate point to covariance space
          const dx = x - center.x;
          const dy = y - center.y;
          
          const rotatedDx = dx * Math.cos(-covariance.rotation) - dy * Math.sin(-covariance.rotation);
          const rotatedDy = dx * Math.sin(-covariance.rotation) + dy * Math.cos(-covariance.rotation);
          
          // Scale by covariance
          const scaledDx = rotatedDx / covariance.scaleX;
          const scaledDy = rotatedDy / covariance.scaleY;
          
          // Calculate Mahalanobis distance
          return Math.sqrt(scaledDx * scaledDx + scaledDy * scaledDy);
        });
        
        // Find the class with minimum distance
        const minDistance = Math.min(...distances);
        const classIdx = distances.indexOf(minDistance);
        
        row.push({ x, y, class: classIdx, distance: minDistance });
      }
      
      grid.push(row);
    }
    
    // Find boundary points (where class changes between adjacent grid cells)
    const boundaryPoints = [];
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const current = grid[i][j];
        const right = grid[i+1]?.[j];
        const down = grid[i][j+1];
        
        if (right && current.class !== right.class) {
          boundaryPoints.push({
            x: (current.x + right.x) / 2,
            y: (current.y + right.y) / 2
          });
        }
        
        if (down && current.class !== down.class) {
          boundaryPoints.push({
            x: (current.x + down.x) / 2,
            y: (current.y + down.y) / 2
          });
        }
      }
    }
    
    // Draw boundary with smooth interpolation
    if (boundaryPoints.length > 0) {
      ctx.beginPath();
      ctx.moveTo(toCanvasX(boundaryPoints[0].x), toCanvasY(boundaryPoints[0].y));
      
      for (let i = 1; i < boundaryPoints.length; i++) {
        ctx.lineTo(toCanvasX(boundaryPoints[i].x), toCanvasY(boundaryPoints[i].y));
      }
      
      ctx.stroke();
    }
    
    ctx.restore();
  };
  
  // Draw probability contours
  const drawProbabilityContours = (progress = 1) => {
    if (!params.show_probability) return;
    
    ctx.save();
    
    // Create a grid of probability values
    const resolution = 50;
    const cellWidth = (width - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        // Calculate probability for each class
        const probabilities = centers.map((center, idx) => {
          const covariance = covariances[idx];
          
          // Calculate Mahalanobis distance
          const dx = x - center.x;
          const dy = y - center.y;
          
          const rotatedDx = dx * Math.cos(-covariance.rotation) - dy * Math.sin(-covariance.rotation);
          const rotatedDy = dx * Math.sin(-covariance.rotation) + dy * Math.cos(-covariance.rotation);
          
          const scaledDx = rotatedDx / covariance.scaleX;
          const scaledDy = rotatedDy / covariance.scaleY;
          
          const distanceSq = scaledDx * scaledDx + scaledDy * scaledDy;
          
          // Gaussian probability (simplified)
          return Math.exp(-distanceSq / 2);
        });
        
        // Normalize probabilities
        const sum = probabilities.reduce((a, b) => a + b, 0);
        const normalized = probabilities.map(p => p / sum);
        
        // Find the class with highest probability
        const maxProb = Math.max(...normalized);
        const classIdx = normalized.indexOf(maxProb);
        const color = COLORS.spectrum[classIdx % COLORS.spectrum.length];
        
        // Map probability to opacity
        const opacity = MathUtils.map(maxProb, 0.5, 1, 0.1, 0.4) * progress;
        
        ctx.fillStyle = color + Math.floor(opacity * 255).toString(16).padStart(2, '0');
        ctx.fillRect(
          toCanvasX(x) - cellWidth / 2,
          toCanvasY(y) - cellHeight / 2,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Animate the QDA visualization with enhanced cinematic effects
  const animateQDA = () => {
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing class by class
    const classes = [...new Set(data.map(p => p.label))];
    
    classes.forEach((classLabel, classIndex) => {
      timeline.add({
        duration: 1000,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points from previous classes
          for (let i = 0; i < classIndex; i++) {
            const classPoints = data.filter(p => p.label === classes[i]);
            drawDataPoints(classPoints, 1);
          }
          
          // Draw points from current class with animation
          const classPoints = data.filter(p => p.label === classLabel);
          drawDataPoints(classPoints, progress);
          
          // Add class label
          if (progress > 0.8) {
            const center = centers[classLabel];
            ctx.save();
            ctx.fillStyle = COLORS.text;
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`Class ${classLabel}`, toCanvasX(center.x), toCanvasY(center.y) - 20);
            ctx.restore();
          }
        }
      }, { delay: classIndex * 300 });
    });
    
    // Phase 2: Animate Gaussian ellipses forming
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 1);
        drawGaussianEllipses(progress);
      }
    });
    
    // Phase 3: Animate decision boundary and probability contours
    timeline.add({
      duration: 1500,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 1);
        drawGaussianEllipses(1);
        
        if (params.show_probability) {
          drawProbabilityContours(progress);
        }
        
        if (params.show_boundary) {
          drawDecisionBoundary(progress);
        }
      }
    });
    
    // Phase 4: Final reveal with all elements
    timeline.add({
      duration: 1000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(data, 1);
        drawGaussianEllipses(1);
        
        if (params.show_probability) {
          drawProbabilityContours(1);
        }
        
        if (params.show_boundary) {
          drawDecisionBoundary(1);
        }
        
        // Draw title and info
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Quadratic Discriminant Analysis', 60, 30);
        ctx.fillText(`Classes: ${params.n_classes}`, 60, 50);
        ctx.fillText('Quadratic Decision Boundary', 60, 70);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'decision-boundary':
      params.show_probability = false;
      params.show_comparison = false;
      animateQDA();
      break;
      
    case 'probability-contours':
      params.show_boundary = false;
      params.show_comparison = false;
      params.show_probability = true;
      animateQDA();
      break;
      
    case 'comparison-lda':
      // This would require implementing LDA comparison
      params.show_boundary = true;
      params.show_probability = false;
      params.show_comparison = true;
      animateQDA();
      break;
      
    case 'all':
      params.show_boundary = true;
      params.show_probability = true;
      params.show_comparison = false;
      animateQDA();
      break;
      
    default:
      animateQDA();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Samples',
        type: 'range',
        min: 50,
        max: 500,
        step: 10,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizers['qda'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Number of Classes',
        type: 'select',
        options: [
          { value: 2, label: '2 Classes', selected: params.n_classes === 2 },
          { value: 3, label: '3 Classes', selected: params.n_classes === 3 },
          { value: 4, label: '4 Classes', selected: params.n_classes === 4 }
        ],
        onChange: (value) => {
          params.n_classes = parseInt(value);
          visualizers['qda'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Covariance Scale',
        type: 'range',
        min: 0.5,
        max: 3,
        step: 0.1,
        value: params.covariance_scale,
        onChange: (value) => {
          params.covariance_scale = parseFloat(value);
          visualizers['qda'](containerId, visualizationType, params);
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
          visualizers['qda'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision Boundary',
        type: 'checkbox',
        checked: params.show_boundary,
        onChange: (value) => {
          params.show_boundary = value;
          visualizers['qda'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Probability Contours',
        type: 'checkbox',
        checked: params.show_probability,
        onChange: (value) => {
          params.show_probability = value;
          visualizers['qda'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'decision-boundary', label: 'Decision Boundary', selected: visualizationType === 'decision-boundary' },
          { value: 'probability-contours', label: 'Probability Contours', selected: visualizationType === 'probability-contours' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizers['qda'](containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'QDA Parameters',
      description: 'Adjust parameters to see how they affect quadratic discriminant analysis.'
    });
  }
};

// =============================================
// Linear Discriminant Analysis Visualization
// =============================================
function visualizeLDA(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 200,
    n_classes: 3,
    n_features: 2,
    class_separation: 1.5,
    show_projection: true,
    show_boundary: false,
    show_comparison: false,
    animation_duration: 3000,
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
  
  // Generate LDA data with Gaussian distributions
  const generateLdaData = () => {
    const data = [];
    const centers = [];
    
    // Create class centers with separation
    for (let i = 0; i < params.n_classes; i++) {
      const angle = (i / params.n_classes) * Math.PI * 2;
      const distance = params.class_separation * 2;
      centers.push({
        x: Math.cos(angle) * distance,
        y: Math.sin(angle) * distance
      });
    }
    
    // Generate samples for each class
    const samplesPerClass = Math.floor(params.n_samples / params.n_classes);
    
    for (let i = 0; i < params.n_classes; i++) {
      const center = centers[i];
      
      for (let j = 0; j < samplesPerClass; j++) {
        // Generate point from normal distribution
        const x = center.x + MathUtils.gaussianRandom(0, 1);
        const y = center.y + MathUtils.gaussianRandom(0, 1);
        
        data.push({ x, y, label: i });
      }
    }
    
    return { data, centers };
  };
  
  const { data, centers } = generateLdaData();
  
  // Find data bounds for scaling
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const padding = 2;
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
  
  // Enhanced data points drawing with class-by-class animation
  const drawDataPoints = (points, progress = 1, highlight = null) => {
    ctx.save();
    
    // Group points by class for sequential animation
    const pointsByClass = {};
    points.forEach(point => {
      if (!pointsByClass[point.label]) pointsByClass[point.label] = [];
      pointsByClass[point.label].push(point);
    });
    
    // Draw each class with different timing
    Object.keys(pointsByClass).forEach((classLabel, classIndex) => {
      const classPoints = pointsByClass[classLabel];
      const classProgress = MathUtils.clamp((progress - classIndex * 0.2) * 1.25, 0, 1);
      
      classPoints.forEach(point => {
        // Dim points that are not in the current highlight
        let alpha = classProgress;
        if (highlight && !highlight.includes(point)) {
          alpha *= 0.3;
        }
        
        ctx.globalAlpha = alpha;
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
    });
    
    ctx.restore();
  };
  
  // Calculate LDA projection direction
  const calculateLdaDirection = () => {
    // Calculate class means
    const classMeans = [];
    const classCounts = [];
    
    for (let i = 0; i < params.n_classes; i++) {
      const classPoints = data.filter(p => p.label === i);
      classCounts.push(classPoints.length);
      
      const meanX = classPoints.reduce((sum, p) => sum + p.x, 0) / classPoints.length;
      const meanY = classPoints.reduce((sum, p) => sum + p.y, 0) / classPoints.length;
      classMeans.push({ x: meanX, y: meanY });
    }
    
    // Calculate overall mean
    const overallMean = {
      x: data.reduce((sum, p) => sum + p.x, 0) / data.length,
      y: data.reduce((sum, p) => sum + p.y, 0) / data.length
    };
    
    // Calculate between-class scatter matrix
    let sb11 = 0, sb12 = 0, sb22 = 0;
    
    for (let i = 0; i < params.n_classes; i++) {
      const dx = classMeans[i].x - overallMean.x;
      const dy = classMeans[i].y - overallMean.y;
      
      sb11 += classCounts[i] * dx * dx;
      sb12 += classCounts[i] * dx * dy;
      sb22 += classCounts[i] * dy * dy;
    }
    
    // Calculate within-class scatter matrix
    let sw11 = 0, sw12 = 0, sw22 = 0;
    
    for (let i = 0; i < params.n_classes; i++) {
      const classPoints = data.filter(p => p.label === i);
      
      classPoints.forEach(point => {
        const dx = point.x - classMeans[i].x;
        const dy = point.y - classMeans[i].y;
        
        sw11 += dx * dx;
        sw12 += dx * dy;
        sw22 += dy * dy;
      });
    }
    
    // Solve generalized eigenvalue problem: Sb * w = λ * Sw * w
    // For simplicity, we'll use a direct approach for 2D case
    const det = sw11 * sw22 - sw12 * sw12;
    const a = (sb11 * sw22 - sb12 * sw12) / det;
    const b = (sb12 * sw11 - sb11 * sw12) / det;
    const c = (sb12 * sw22 - sb22 * sw12) / det;
    const d = (sb22 * sw11 - sb12 * sw12) / det;
    
    // Find eigenvector corresponding to largest eigenvalue
    const trace = a + d;
    const discriminant = Math.sqrt(trace * trace - 4 * (a * d - b * c));
    const eigenvalue1 = (trace + discriminant) / 2;
    const eigenvalue2 = (trace - discriminant) / 2;
    
    // Use eigenvector for largest eigenvalue
    let directionX, directionY;
    
    if (eigenvalue1 > eigenvalue2) {
      directionX = b;
      directionY = eigenvalue1 - a;
    } else {
      directionX = b;
      directionY = eigenvalue2 - a;
    }
    
    // Normalize direction vector
    const magnitude = Math.sqrt(directionX * directionX + directionY * directionY);
    directionX /= magnitude;
    directionY /= magnitude;
    
    return { directionX, directionY, classMeans, overallMean };
  };
  
  const { directionX, directionY, classMeans, overallMean } = calculateLdaDirection();
  
  // Draw LDA projection direction
  const drawProjectionDirection = (progress = 1) => {
    if (!params.show_projection) return;
    
    ctx.save();
    ctx.strokeStyle = COLORS.highlight;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.globalAlpha = progress;
    
    // Calculate line endpoints along the projection direction
    const scale = 10;
    const x1 = overallMean.x - directionX * scale;
    const y1 = overallMean.y - directionY * scale;
    const x2 = overallMean.x + directionX * scale;
    const y2 = overallMean.y + directionY * scale;
    
    ctx.beginPath();
    ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
    ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
    ctx.stroke();
    
    // Draw arrowhead
    const arrowSize = 10;
    const angle = Math.atan2(directionY, directionX);
    
    ctx.beginPath();
    ctx.moveTo(toCanvasX(x2), toCanvasY(y2));
    ctx.lineTo(
      toCanvasX(x2 - arrowSize * Math.cos(angle - Math.PI/6)),
      toCanvasY(y2 - arrowSize * Math.sin(angle - Math.PI/6))
    );
    ctx.lineTo(
      toCanvasX(x2 - arrowSize * Math.cos(angle + Math.PI/6)),
      toCanvasY(y2 - arrowSize * Math.sin(angle + Math.PI/6))
    );
    ctx.closePath();
    ctx.fillStyle = COLORS.highlight;
    ctx.fill();
    
    // Draw projection label
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('LDA Projection Direction', toCanvasX(overallMean.x), toCanvasY(overallMean.y) - 20);
    
    ctx.restore();
  };
  
  // Draw projected points
  const drawProjectedPoints = (progress = 1) => {
    if (!params.show_projection) return;
    
    ctx.save();
    
    // Calculate projection of each point onto LDA direction
    data.forEach(point => {
      // Vector from overall mean to point
      const dx = point.x - overallMean.x;
      const dy = point.y - overallMean.y;
      
      // Project onto LDA direction
      const projection = dx * directionX + dy * directionY;
      
      // Calculate projected point position
      const projX = overallMean.x + projection * directionX;
      const projY = overallMean.y + projection * directionY;
      
      // Draw line from point to projection
      ctx.globalAlpha = 0.3 * progress;
      ctx.strokeStyle = COLORS.spectrum[point.label % COLORS.spectrum.length];
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      
      ctx.beginPath();
      ctx.moveTo(toCanvasX(point.x), toCanvasY(point.y));
      ctx.lineTo(toCanvasX(projX), toCanvasY(projY));
      ctx.stroke();
      
      // Draw projected point
      ctx.globalAlpha = progress;
      ctx.fillStyle = COLORS.spectrum[point.label % COLORS.spectrum.length];
      ctx.beginPath();
      ctx.arc(toCanvasX(projX), toCanvasY(projY), 4, 0, Math.PI * 2);
      ctx.fill();
    });
    
    ctx.setLineDash([]);
    ctx.restore();
  };
  
  // Draw linear decision boundaries
  const drawDecisionBoundaries = (progress = 1) => {
    if (!params.show_boundary) return;
    
    ctx.save();
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.globalAlpha = progress;
    
    // For LDA, boundaries are lines perpendicular to the projection direction
    // Calculate boundary positions between class means
    
    for (let i = 0; i < params.n_classes - 1; i++) {
      for (let j = i + 1; j < params.n_classes; j++) {
        // Calculate midpoint between class means
        const midX = (classMeans[i].x + classMeans[j].x) / 2;
        const midY = (classMeans[i].y + classMeans[j].y) / 2;
        
        // Calculate boundary line (perpendicular to projection direction)
        const boundaryLength = 20;
        const perpX = -directionY;
        const perpY = directionX;
        
        const x1 = midX - perpX * boundaryLength;
        const y1 = midY - perpY * boundaryLength;
        const x2 = midX + perpX * boundaryLength;
        const y2 = midY + perpY * boundaryLength;
        
        ctx.beginPath();
        ctx.moveTo(toCanvasX(x1), toCanvasY(y1));
        ctx.lineTo(toCanvasX(x2), toCanvasY(y2));
        ctx.stroke();
      }
    }
    
    ctx.restore();
  };
  
  // Animate the LDA visualization with enhanced cinematic effects
  const animateLDA = () => {
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing class by class
    const classes = [...new Set(data.map(p => p.label))];
    
    classes.forEach((classLabel, classIndex) => {
      timeline.add({
        duration: 1000,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points from previous classes
          for (let i = 0; i < classIndex; i++) {
            const classPoints = data.filter(p => p.label === classes[i]);
            drawDataPoints(classPoints, 1);
          }
          
          // Draw points from current class with animation
          const classPoints = data.filter(p => p.label === classLabel);
          drawDataPoints(classPoints, progress);
          
          // Add class label
          if (progress > 0.8) {
            const center = centers[classLabel];
            ctx.save();
            ctx.fillStyle = COLORS.text;
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`Class ${classLabel}`, toCanvasX(center.x), toCanvasY(center.y) - 20);
            ctx.restore();
          }
        }
      }, { delay: classIndex * 300 });
    });
    
    // Phase 2: Animate class means and overall mean
    timeline.add({
      duration: 1000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 1);
        
        // Draw class means
        classMeans.forEach((mean, i) => {
          ctx.save();
          ctx.globalAlpha = progress;
          ctx.fillStyle = COLORS.spectrum[i % COLORS.spectrum.length];
          ctx.beginPath();
          ctx.arc(toCanvasX(mean.x), toCanvasY(mean.y), 8, 0, Math.PI * 2);
          ctx.fill();
          ctx.strokeStyle = COLORS.text;
          ctx.lineWidth = 2;
          ctx.stroke();
          ctx.restore();
        });
        
        // Draw overall mean
        ctx.save();
        ctx.globalAlpha = progress;
        ctx.fillStyle = COLORS.highlight;
        ctx.beginPath();
        ctx.arc(toCanvasX(overallMean.x), toCanvasY(overallMean.y), 10, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.restore();
      }
    });
    
    // Phase 3: Animate LDA projection direction
    timeline.add({
      duration: 1000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 1);
        drawProjectionDirection(progress);
      }
    });
    
    // Phase 4: Animate projection lines and projected points
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 1);
        drawProjectionDirection(1);
        drawProjectedPoints(progress);
      }
    });
    
    // Phase 5: Animate decision boundaries if enabled
    if (params.show_boundary) {
      timeline.add({
        duration: 1000,
        easing: 'easeOutCubic',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          drawDataPoints(data, 1);
          drawProjectionDirection(1);
          drawProjectedPoints(1);
          drawDecisionBoundaries(progress);
        }
      });
    }
    
    // Phase 6: Final reveal with all elements
    timeline.add({
      duration: 1000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(data, 1);
        drawProjectionDirection(1);
        
        if (params.show_projection) {
          drawProjectedPoints(1);
        }
        
        if (params.show_boundary) {
          drawDecisionBoundaries(1);
        }
        
        // Draw title and info
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Linear Discriminant Analysis', 60, 30);
        ctx.fillText(`Classes: ${params.n_classes}`, 60, 50);
        ctx.fillText('Optimal Projection Direction', 60, 70);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'projection':
      params.show_boundary = false;
      params.show_comparison = false;
      params.show_projection = true;
      animateLDA();
      break;
      
    case 'decision-boundary':
      params.show_projection = false;
      params.show_comparison = false;
      params.show_boundary = true;
      animateLDA();
      break;
      
    case 'comparison-pca':
      // This would require implementing PCA comparison
      params.show_projection = true;
      params.show_boundary = false;
      params.show_comparison = true;
      animateLDA();
      break;
      
    case 'all':
      params.show_projection = true;
      params.show_boundary = true;
      params.show_comparison = false;
      animateLDA();
      break;
      
    default:
      animateLDA();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Samples',
        type: 'range',
        min: 50,
        max: 500,
        step: 10,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizers['lda'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Number of Classes',
        type: 'select',
        options: [
          { value: 2, label: '2 Classes', selected: params.n_classes === 2 },
          { value: 3, label: '3 Classes', selected: params.n_classes === 3 },
          { value: 4, label: '4 Classes', selected: params.n_classes === 4 }
        ],
        onChange: (value) => {
          params.n_classes = parseInt(value);
          visualizers['lda'](containerId, visualizationType, params);
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
          visualizers['lda'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Projection',
        type: 'checkbox',
        checked: params.show_projection,
        onChange: (value) => {
          params.show_projection = value;
          visualizers['lda'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision Boundaries',
        type: 'checkbox',
        checked: params.show_boundary,
        onChange: (value) => {
          params.show_boundary = value;
          visualizers['lda'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'projection', label: 'Projection View', selected: visualizationType === 'projection' },
          { value: 'decision-boundary', label: 'Decision Boundary', selected: visualizationType === 'decision-boundary' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizers['lda'](containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'LDA Parameters',
      description: 'Adjust parameters to see how they affect linear discriminant analysis.'
    });
  }
};

// =============================================
// Naive Bayes Visualization
// =============================================
function visualizeNaiveBayes(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 150,
    n_classes: 2,
    distribution_type: 'gaussian',
    class_separation: 1.0,
    show_distributions: true,
    show_boundary: true,
    show_probability: false,
    show_priors: false,
    animation_duration: 3000,
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
  
  // Generate Naive Bayes data with Gaussian distributions
  const generateNaiveBayesData = () => {
    const data = [];
    const centers = [];
    const priors = [];
    
    // Create class centers with separation
    for (let i = 0; i < params.n_classes; i++) {
      const angle = (i / params.n_classes) * Math.PI * 2;
      const distance = params.class_separation * 2;
      centers.push({
        x: Math.cos(angle) * distance,
        y: Math.sin(angle) * distance
      });
      
      // Create priors (sum to 1)
      priors.push(1 / params.n_classes);
    }
    
    // Generate samples for each class
    const samplesPerClass = params.n_samples / params.n_classes;
    
    for (let i = 0; i < params.n_classes; i++) {
      const center = centers[i];
      const classSamples = Math.floor(samplesPerClass * priors[i]);
      
      for (let j = 0; j < classSamples; j++) {
        // Generate point from normal distribution
        const x = center.x + MathUtils.gaussianRandom(0, 1);
        const y = center.y + MathUtils.gaussianRandom(0, 1);
        
        data.push({ x, y, label: i });
      }
    }
    
    return { data, centers, priors };
  };
  
  const { data, centers, priors } = generateNaiveBayesData();
  
  // Find data bounds for scaling
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const padding = 2;
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
  
  // Enhanced data points drawing with class-by-class animation
  const drawDataPoints = (points, progress = 1) => {
    ctx.save();
    
    // Group points by class for sequential animation
    const pointsByClass = {};
    points.forEach(point => {
      if (!pointsByClass[point.label]) pointsByClass[point.label] = [];
      pointsByClass[point.label].push(point);
    });
    
    // Draw each class with different timing
    Object.keys(pointsByClass).forEach((classLabel, classIndex) => {
      const classPoints = pointsByClass[classLabel];
      const classProgress = MathUtils.clamp((progress - classIndex * 0.2) * 1.25, 0, 1);
      
      classPoints.forEach(point => {
        ctx.globalAlpha = classProgress;
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
    });
    
    ctx.restore();
  };
  
  // Calculate class statistics for Naive Bayes
  const calculateClassStats = () => {
    const stats = [];
    
    for (let i = 0; i < params.n_classes; i++) {
      const classPoints = data.filter(p => p.label === i);
      const meanX = classPoints.reduce((sum, p) => sum + p.x, 0) / classPoints.length;
      const meanY = classPoints.reduce((sum, p) => sum + p.y, 0) / classPoints.length;
      
      // Calculate variance (Naive Bayes assumes independence between features)
      const varX = classPoints.reduce((sum, p) => sum + Math.pow(p.x - meanX, 2), 0) / classPoints.length;
      const varY = classPoints.reduce((sum, p) => sum + Math.pow(p.y - meanY, 2), 0) / classPoints.length;
      
      stats.push({
        meanX,
        meanY,
        varX,
        varY,
        count: classPoints.length
      });
    }
    
    return stats;
  };
  
  const classStats = calculateClassStats();
  
  // Draw Gaussian distributions for each class
  const drawDistributions = (progress = 1) => {
    if (!params.show_distributions) return;
    
    ctx.save();
    
    // Create a grid of probability values
    const resolution = 50;
    const cellWidth = (width - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        // Calculate probability for each class
        const probabilities = classStats.map((stats, idx) => {
          // Gaussian probability (Naive Bayes assumes independence)
          const pX = Math.exp(-Math.pow(x - stats.meanX, 2) / (2 * stats.varX)) / Math.sqrt(2 * Math.PI * stats.varX);
          const pY = Math.exp(-Math.pow(y - stats.meanY, 2) / (2 * stats.varY)) / Math.sqrt(2 * Math.PI * stats.varY);
          
          // Joint probability (assuming independence)
          return pX * pY * priors[idx];
        });
        
        // Find the class with highest probability
        const maxProb = Math.max(...probabilities);
        const classIdx = probabilities.indexOf(maxProb);
        const color = COLORS.spectrum[classIdx % COLORS.spectrum.length];
        
        // Map probability to opacity
        const opacity = MathUtils.map(maxProb, 0, Math.max(...probabilities), 0.1, 0.6) * progress;
        
        ctx.fillStyle = color + Math.floor(opacity * 255).toString(16).padStart(2, '0');
        ctx.fillRect(
          toCanvasX(x) - cellWidth / 2,
          toCanvasY(y) - cellHeight / 2,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Draw decision boundary
  const drawDecisionBoundary = (progress = 1) => {
    if (!params.show_boundary) return;
    
    ctx.save();
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.globalAlpha = progress;
    
    // For Naive Bayes, we'll approximate the boundary by sampling points
    // where the difference between class probabilities is minimal
    
    // Create a grid of points
    const resolution = 100;
    const grid = [];
    
    for (let i = 0; i <= resolution; i++) {
      const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
      const row = [];
      
      for (let j = 0; j <= resolution; j++) {
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        // Calculate probability for each class
        const probabilities = classStats.map((stats, idx) => {
          // Gaussian probability (Naive Bayes assumes independence)
          const pX = Math.exp(-Math.pow(x - stats.meanX, 2) / (2 * stats.varX)) / Math.sqrt(2 * Math.PI * stats.varX);
          const pY = Math.exp(-Math.pow(y - stats.meanY, 2) / (2 * stats.varY)) / Math.sqrt(2 * Math.PI * stats.varY);
          
          // Joint probability (assuming independence)
          return pX * pY * priors[idx];
        });
        
        // Find the class with maximum probability
        const maxProb = Math.max(...probabilities);
        const classIdx = probabilities.indexOf(maxProb);
        
        row.push({ x, y, class: classIdx, probability: maxProb });
      }
      
      grid.push(row);
    }
    
    // Find boundary points (where class changes between adjacent grid cells)
    const boundaryPoints = [];
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const current = grid[i][j];
        const right = grid[i+1]?.[j];
        const down = grid[i][j+1];
        
        if (right && current.class !== right.class) {
          boundaryPoints.push({
            x: (current.x + right.x) / 2,
            y: (current.y + right.y) / 2
          });
        }
        
        if (down && current.class !== down.class) {
          boundaryPoints.push({
            x: (current.x + down.x) / 2,
            y: (current.y + down.y) / 2
          });
        }
      }
    }
    
    // Draw boundary with smooth interpolation
    if (boundaryPoints.length > 0) {
      ctx.beginPath();
      ctx.moveTo(toCanvasX(boundaryPoints[0].x), toCanvasY(boundaryPoints[0].y));
      
      for (let i = 1; i < boundaryPoints.length; i++) {
        ctx.lineTo(toCanvasX(boundaryPoints[i].x), toCanvasY(boundaryPoints[i].y));
      }
      
      ctx.stroke();
    }
    
    ctx.restore();
  };
  
  // Draw prior probabilities
  const drawPriors = (progress = 1) => {
    if (!params.show_priors) return;
    
    ctx.save();
    
    // Draw prior probabilities as bars
    const barWidth = 40;
    const barSpacing = 20;
    const maxBarHeight = 100;
    const startX = 100;
    const startY = height - 150;
    
    priors.forEach((prior, i) => {
      const barHeight = prior * maxBarHeight;
      const x = startX + i * (barWidth + barSpacing);
      
      ctx.globalAlpha = progress;
      ctx.fillStyle = COLORS.spectrum[i % COLORS.spectrum.length];
      ctx.fillRect(x, startY - barHeight, barWidth, barHeight);
      
      // Draw bar outline
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1;
      ctx.strokeRect(x, startY - barHeight, barWidth, barHeight);
      
      // Draw label
      ctx.fillStyle = COLORS.text;
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`P(C${i}) = ${prior.toFixed(2)}`, x + barWidth / 2, startY + 20);
    });
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Prior Probabilities', startX + (priors.length * (barWidth + barSpacing)) / 2, startY - maxBarHeight - 20);
    
    ctx.restore();
  };
  
  // Animate the Naive Bayes visualization with enhanced cinematic effects
  const animateNaiveBayes = () => {
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing class by class
    const classes = [...new Set(data.map(p => p.label))];
    
    classes.forEach((classLabel, classIndex) => {
      timeline.add({
        duration: 1000,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points from previous classes
          for (let i = 0; i < classIndex; i++) {
            const classPoints = data.filter(p => p.label === classes[i]);
            drawDataPoints(classPoints, 1);
          }
          
          // Draw points from current class with animation
          const classPoints = data.filter(p => p.label === classLabel);
          drawDataPoints(classPoints, progress);
          
          // Add class label
          if (progress > 0.8) {
            const center = centers[classLabel];
            ctx.save();
            ctx.fillStyle = COLORS.text;
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`Class ${classLabel}`, toCanvasX(center.x), toCanvasY(center.y) - 20);
            ctx.restore();
          }
        }
      }, { delay: classIndex * 300 });
    });
    
    // Phase 2: Animate prior probabilities if enabled
    if (params.show_priors) {
      timeline.add({
        duration: 1000,
        easing: 'easeOutCubic',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          drawDataPoints(data, 1);
          drawPriors(progress);
        }
      });
    }
    
    // Phase 3: Animate Gaussian distributions forming
    if (params.show_distributions) {
      timeline.add({
        duration: 1500,
        easing: 'easeOutCubic',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          drawDataPoints(data, 1);
          if (params.show_priors) drawPriors(1);
          drawDistributions(progress);
        }
      });
    }
    
    // Phase 4: Animate decision boundary
    if (params.show_boundary) {
      timeline.add({
        duration: 1000,
        easing: 'easeOutCubic',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          drawDataPoints(data, 1);
          if (params.show_priors) drawPriors(1);
          if (params.show_distributions) drawDistributions(1);
          drawDecisionBoundary(progress);
        }
      });
    }
    
    // Phase 5: Final reveal with all elements
    timeline.add({
      duration: 1000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(data, 1);
        if (params.show_priors) drawPriors(1);
        if (params.show_distributions) drawDistributions(1);
        if (params.show_boundary) drawDecisionBoundary(1);
        
        // Draw title and info
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Naive Bayes Classifier', 60, 30);
        ctx.fillText(`Classes: ${params.n_classes}`, 60, 50);
        ctx.fillText('Gaussian Distributions', 60, 70);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'probability-surface':
      params.show_distributions = true;
      params.show_boundary = false;
      params.show_priors = false;
      animateNaiveBayes();
      break;
      
    case 'feature-distributions':
      params.show_distributions = true;
      params.show_boundary = false;
      params.show_priors = true;
      animateNaiveBayes();
      break;
      
    case 'decision-boundary':
      params.show_distributions = false;
      params.show_boundary = true;
      params.show_priors = false;
      animateNaiveBayes();
      break;
      
    case 'all':
      params.show_distributions = true;
      params.show_boundary = true;
      params.show_priors = true;
      animateNaiveBayes();
      break;
      
    case 'with-priors':
      params.show_distributions = true;
      params.show_boundary = true;
      params.show_priors = true;
      animateNaiveBayes();
      break;
      
    default:
      animateNaiveBayes();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Samples',
        type: 'range',
        min: 50,
        max: 500,
        step: 10,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizers['naive-bayes'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Number of Classes',
        type: 'select',
        options: [
          { value: 2, label: '2 Classes', selected: params.n_classes === 2 },
          { value: 3, label: '3 Classes', selected: params.n_classes === 3 },
          { value: 4, label: '4 Classes', selected: params.n_classes === 4 }
        ],
        onChange: (value) => {
          params.n_classes = parseInt(value);
          visualizers['naive-bayes'](containerId, visualizationType, params);
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
          visualizers['naive-bayes'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Distributions',
        type: 'checkbox',
        checked: params.show_distributions,
        onChange: (value) => {
          params.show_distributions = value;
          visualizers['naive-bayes'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Decision Boundary',
        type: 'checkbox',
        checked: params.show_boundary,
        onChange: (value) => {
          params.show_boundary = value;
          visualizers['naive-bayes'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Priors',
        type: 'checkbox',
        checked: params.show_priors,
        onChange: (value) => {
          params.show_priors = value;
          visualizers['naive-bayes'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'probability-surface', label: 'Probability Surface', selected: visualizationType === 'probability-surface' },
          { value: 'feature-distributions', label: 'Feature Distributions', selected: visualizationType === 'feature-distributions' },
          { value: 'decision-boundary', label: 'Decision Boundary', selected: visualizationType === 'decision-boundary' },
          { value: 'with-priors', label: 'With Priors', selected: visualizationType === 'with-priors' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizers['naive-bayes'](containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'Naive Bayes Parameters',
      description: 'Adjust parameters to see how they affect Naive Bayes classification.'
    });
  }
};

// =============================================
// K-Means Clustering Visualization
// =============================================
function visualizeKMeans(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 300,
    n_clusters: 3,
    cluster_std: 0.8,
    init_method: 'k-means++',
    max_iter: 10,
    show_centroids: true,
    show_assignments: true,
    show_boundaries: false,
    show_elbow: false,
    animation_duration: 3000,
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
  
  // Generate clustering data
  const data = DataSimulator.generateClusteringData({
    n_samples: params.n_samples,
    n_clusters: params.n_clusters,
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
  
  // Enhanced data points drawing with cluster assignment animation
  const drawDataPoints = (points, centroids = [], assignments = [], progress = 1) => {
    ctx.save();
    
    points.forEach((point, i) => {
      let color;
      let alpha = progress;
      
      if (assignments.length > 0 && centroids.length > 0) {
        // Use cluster color if assigned
        const clusterIdx = assignments[i];
        color = COLORS.spectrum[clusterIdx % COLORS.spectrum.length];
        
        // Dim points that are far from their centroid
        if (centroids[clusterIdx]) {
          const distance = MathUtils.distance(point.x, point.y, centroids[clusterIdx].x, centroids[clusterIdx].y);
          const maxDistance = 5; // Adjust based on data scale
          alpha *= MathUtils.clamp(1 - (distance / maxDistance) * 0.5, 0.5, 1);
        }
      } else {
        // Use true cluster color if no assignments yet
        color = COLORS.spectrum[point.trueCluster % COLORS.spectrum.length];
      }
      
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 5, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    
    ctx.restore();
  };
  
  // Draw centroids with pulsing animation
  const drawCentroids = (centroids, progress = 1, isMoving = false) => {
    if (!params.show_centroids) return;
    
    ctx.save();
    
    centroids.forEach((centroid, i) => {
      const color = COLORS.spectrum[i % COLORS.spectrum.length];
      
      // Pulsing effect for centroids
      const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 300);
      const size = isMoving ? 12 + 4 * pulse : 10 + 2 * pulse;
      
      ctx.globalAlpha = progress;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(toCanvasX(centroid.x), toCanvasY(centroid.y), size, 0, Math.PI * 2);
      ctx.fill();
      
      // Highlight moving centroids
      if (isMoving) {
        ctx.strokeStyle = COLORS.highlight;
        ctx.lineWidth = 3;
        ctx.stroke();
      } else {
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 2;
        ctx.stroke();
      }
      
      // Draw centroid label
      ctx.fillStyle = COLORS.text;
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`C${i}`, toCanvasX(centroid.x), toCanvasY(centroid.y) - 20);
    });
    
    ctx.restore();
  };
  
  // Draw assignment lines from points to centroids
  const drawAssignmentLines = (points, centroids, assignments, progress = 1) => {
    if (!params.show_assignments) return;
    
    ctx.save();
    
    points.forEach((point, i) => {
      if (assignments[i] !== undefined && centroids[assignments[i]]) {
        const centroid = centroids[assignments[i]];
        const color = COLORS.spectrum[assignments[i] % COLORS.spectrum.length];
        
        ctx.globalAlpha = 0.3 * progress;
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        
        ctx.beginPath();
        ctx.moveTo(toCanvasX(point.x), toCanvasY(point.y));
        ctx.lineTo(toCanvasX(centroid.x), toCanvasY(centroid.y));
        ctx.stroke();
      }
    });
    
    ctx.setLineDash([]);
    ctx.restore();
  };
  
  // Draw Voronoi diagram boundaries
  const drawVoronoiBoundaries = (centroids, progress = 1) => {
    if (!params.show_boundaries || centroids.length < 2) return;
    
    ctx.save();
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.6 * progress;
    ctx.setLineDash([5, 5]);
    
    // Simple Voronoi approximation by drawing perpendicular bisectors
    for (let i = 0; i < centroids.length; i++) {
      for (let j = i + 1; j < centroids.length; j++) {
        const c1 = centroids[i];
        const c2 = centroids[j];
        
        // Midpoint
        const midX = (c1.x + c2.x) / 2;
        const midY = (c1.y + c2.y) / 2;
        
        // Slope of line between centroids
        const dx = c2.x - c1.x;
        const dy = c2.y - c1.y;
        
        if (Math.abs(dx) < 0.001 && Math.abs(dy) < 0.001) continue;
        
        // Slope of perpendicular bisector
        const slope = -dx / dy;
        
        // Draw a segment of the bisector
        const length = 20;
        const angle = Math.atan(slope);
        
        ctx.beginPath();
        ctx.moveTo(toCanvasX(midX - length * Math.cos(angle)), toCanvasY(midY - length * Math.sin(angle)));
        ctx.lineTo(toCanvasX(midX + length * Math.cos(angle)), toCanvasY(midY + length * Math.sin(angle)));
        ctx.stroke();
      }
    }
    
    ctx.setLineDash([]);
    ctx.restore();
  };
  
  // Draw elbow method chart
  const drawElbowChart = (inertiaValues, progress = 1) => {
    if (!params.show_elbow || inertiaValues.length === 0) return;
    
    ctx.save();
    
    // Create a separate area for the elbow chart
    const chartWidth = 300;
    const chartHeight = 200;
    const chartX = width - chartWidth - 20;
    const chartY = 20;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(chartX, chartY, chartWidth, chartHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(chartX, chartY, chartWidth, chartHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Elbow Method', chartX + chartWidth / 2, chartY - 10);
    
    // Find min and max inertia for scaling
    const maxInertia = Math.max(...inertiaValues);
    const minInertia = Math.min(...inertiaValues);
    const inertiaRange = maxInertia - minInertia || 1;
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(chartX + 30, chartY + 20);
    ctx.lineTo(chartX + 30, chartY + chartHeight - 20);
    ctx.lineTo(chartX + chartWidth - 20, chartY + chartHeight - 20);
    ctx.stroke();
    
    // Draw labels
    ctx.textAlign = 'right';
    ctx.fillText(minInertia.toFixed(0), chartX + 25, chartY + chartHeight - 20);
    ctx.fillText(maxInertia.toFixed(0), chartX + 25, chartY + 20);
    ctx.textAlign = 'center';
    ctx.fillText('Number of Clusters (K)', chartX + chartWidth / 2, chartY + chartHeight);
    ctx.textAlign = 'right';
    ctx.fillText('Inertia', chartX + 25, chartY + chartHeight / 2);
    
    // Draw inertia curve
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    inertiaValues.forEach((inertia, k) => {
      const x = chartX + 30 + (chartWidth - 50) * (k / (inertiaValues.length - 1));
      const y = chartY + chartHeight - 20 - (chartHeight - 40) * ((inertia - minInertia) / inertiaRange);
      
      if (k === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw points
    inertiaValues.forEach((inertia, k) => {
      const x = chartX + 30 + (chartWidth - 50) * (k / (inertiaValues.length - 1));
      const y = chartY + chartHeight - 20 - (chartHeight - 40) * ((inertia - minInertia) / inertiaRange);
      
      ctx.fillStyle = COLORS.accent;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw k value
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`K=${k+1}`, x, y + 15);
    });
    
    // Highlight elbow point (optimal K)
    if (inertiaValues.length > 2) {
      // Simple elbow detection (find point with maximum curvature)
      let maxCurvature = 0;
      let elbowK = 1;
      
      for (let k = 1; k < inertiaValues.length - 1; k++) {
        const prev = inertiaValues[k-1];
        const current = inertiaValues[k];
        const next = inertiaValues[k+1];
        
        const curvature = Math.abs((next - current) - (current - prev));
        if (curvature > maxCurvature) {
          maxCurvature = curvature;
          elbowK = k;
        }
      }
      
      const x = chartX + 30 + (chartWidth - 50) * (elbowK / (inertiaValues.length - 1));
      const y = chartY + chartHeight - 20 - (chartHeight - 40) * ((inertiaValues[elbowK] - minInertia) / inertiaRange);
      
      // Draw highlight
      ctx.fillStyle = COLORS.highlight;
      ctx.beginPath();
      ctx.arc(x, y, 8, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw recommendation
      ctx.fillStyle = COLORS.text;
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`Optimal K: ${elbowK+1}`, chartX + chartWidth / 2, chartY + chartHeight + 20);
    }
    
    ctx.restore();
  };
  
  // K-Means algorithm implementation
  class KMeans {
    constructor(k, data) {
      this.k = k;
      this.data = data;
      this.centroids = [];
      this.assignments = [];
      this.inertia = 0;
      this.iteration = 0;
      this.history = [];
    }
    
    initializeCentroids(method = 'k-means++') {
      this.centroids = [];
      
      if (method === 'random') {
        // Random initialization
        for (let i = 0; i < this.k; i++) {
          const randomIndex = Math.floor(Math.random() * this.data.length);
          this.centroids.push({ 
            x: this.data[randomIndex].x, 
            y: this.data[randomIndex].y 
          });
        }
      } else {
        // k-means++ initialization
        // First centroid is random
        const firstIndex = Math.floor(Math.random() * this.data.length);
        this.centroids.push({ 
          x: this.data[firstIndex].x, 
          y: this.data[firstIndex].y 
        });
        
        // Subsequent centroids are chosen with probability proportional to distance²
        for (let i = 1; i < this.k; i++) {
          const distances = this.data.map(point => {
            const minDistance = Math.min(...this.centroids.map(centroid => 
              MathUtils.distance(point.x, point.y, centroid.x, centroid.y)
            ));
            return minDistance * minDistance;
          });
          
          const sum = distances.reduce((a, b) => a + b, 0);
          let threshold = Math.random() * sum;
          
          for (let j = 0; j < this.data.length; j++) {
            threshold -= distances[j];
            if (threshold <= 0) {
              this.centroids.push({ 
                x: this.data[j].x, 
                y: this.data[j].y 
              });
              break;
            }
          }
        }
      }
    }
    
    assignPoints() {
      this.assignments = [];
      this.inertia = 0;
      
      this.data.forEach(point => {
        let minDistance = Infinity;
        let closestCentroid = -1;
        
        this.centroids.forEach((centroid, i) => {
          const distance = MathUtils.distance(point.x, point.y, centroid.x, centroid.y);
          if (distance < minDistance) {
            minDistance = distance;
            closestCentroid = i;
          }
        });
        
        this.assignments.push(closestCentroid);
        this.inertia += minDistance * minDistance;
      });
    }
    
    updateCentroids() {
      const newCentroids = [];
      const counts = new Array(this.k).fill(0);
      
      // Initialize new centroids
      for (let i = 0; i < this.k; i++) {
        newCentroids.push({ x: 0, y: 0 });
      }
      
      // Sum up points for each cluster
      this.data.forEach((point, i) => {
        const cluster = this.assignments[i];
        newCentroids[cluster].x += point.x;
        newCentroids[cluster].y += point.y;
        counts[cluster]++;
      });
      
      // Calculate means
      for (let i = 0; i < this.k; i++) {
        if (counts[i] > 0) {
          newCentroids[i].x /= counts[i];
          newCentroids[i].y /= counts[i];
        } else {
          // If a cluster has no points, reinitialize it
          const randomIndex = Math.floor(Math.random() * this.data.length);
          newCentroids[i] = { 
            x: this.data[randomIndex].x, 
            y: this.data[randomIndex].y 
          };
        }
      }
      
      // Save history
      this.history.push({
        centroids: [...this.centroids],
        assignments: [...this.assignments],
        inertia: this.inertia
      });
      
      this.centroids = newCentroids;
      this.iteration++;
    }
    
    run(maxIterations = 10) {
      this.initializeCentroids();
      
      for (let i = 0; i < maxIterations; i++) {
        this.assignPoints();
        this.updateCentroids();
      }
      
      // Final assignment
      this.assignPoints();
      this.history.push({
        centroids: [...this.centroids],
        assignments: [...this.assignments],
        inertia: this.inertia
      });
    }
  }
  
  // Animate the K-Means clustering with enhanced cinematic effects
  const animateKMeans = () => {
    const kmeans = new KMeans(params.n_clusters, data);
    const maxIterations = params.max_iter;
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing
    timeline.add({
      duration: 2000,
      easing: 'easeOutBack',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        
        // Draw points that have appeared
        const pointsToShow = Math.floor(progress * data.length);
        drawDataPoints(data.slice(0, pointsToShow));
        
        // Pulse effect for newly appearing points
        if (progress > 0.9) {
          const newestPoint = data[Math.floor(pointsToShow) - 1];
          ctx.save();
          ctx.beginPath();
          ctx.arc(
            toCanvasX(newestPoint.x),
            toCanvasY(newestPoint.y),
            10 * (1 - (progress % 0.1) * 10),
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
    
    // Phase 2: Animate centroid initialization
    timeline.add({
      duration: 1500,
      easing: 'easeOutElastic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data);
        
        // Initialize centroids if not done yet
        if (kmeans.centroids.length === 0) {
          kmeans.initializeCentroids(params.init_method);
        }
        
        // Draw centroids with growing effect
        const centroidsToShow = Math.floor(progress * kmeans.centroids.length);
        drawCentroids(kmeans.centroids.slice(0, centroidsToShow), progress, false);
      }
    });
    
    // Phase 3: Animate iterations
    for (let iter = 0; iter < maxIterations; iter++) {
      // Assignment step
      timeline.add({
        duration: 1500,
        easing: 'easeOutCubic',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Run assignment if not done yet
          if (kmeans.assignments.length === 0 || iter > 0) {
            kmeans.assignPoints();
          }
          
          drawDataPoints(data, kmeans.centroids, kmeans.assignments, progress);
          drawCentroids(kmeans.centroids, 1, false);
          
          if (params.show_assignments) {
            drawAssignmentLines(data, kmeans.centroids, kmeans.assignments, progress);
          }
          
          if (params.show_boundaries) {
            drawVoronoiBoundaries(kmeans.centroids, progress);
          }
          
          // Draw iteration info
          ctx.fillStyle = COLORS.text;
          ctx.font = '16px Arial';
          ctx.textAlign = 'left';
          ctx.fillText(`Iteration: ${iter + 1}/${maxIterations}`, 60, 30);
          ctx.fillText(`Inertia: ${kmeans.inertia.toFixed(2)}`, 60, 50);
        }
      });
      
      // Update step (centroid movement)
      timeline.add({
        duration: 2000,
        easing: 'easeInOutCubic',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Calculate intermediate centroids for smooth animation
          const oldCentroids = kmeans.history.length > 0 ? 
            kmeans.history[kmeans.history.length - 1].centroids : kmeans.centroids;
          
          const newCentroids = [];
          for (let i = 0; i < kmeans.centroids.length; i++) {
            const oldCentroid = oldCentroids[i];
            const newCentroid = kmeans.centroids[i];
            
            newCentroids.push({
              x: MathUtils.lerp(oldCentroid.x, newCentroid.x, progress),
              y: MathUtils.lerp(oldCentroid.y, newCentroid.y, progress)
            });
          }
          
          drawDataPoints(data, newCentroids, kmeans.assignments, 1);
          drawCentroids(newCentroids, 1, true);
          
          if (params.show_assignments) {
            drawAssignmentLines(data, newCentroids, kmeans.assignments, 1);
          }
          
          if (params.show_boundaries) {
            drawVoronoiBoundaries(newCentroids, 1);
          }
          
          // Draw iteration info
          ctx.fillStyle = COLORS.text;
          ctx.font = '16px Arial';
          ctx.textAlign = 'left';
          ctx.fillText(`Iteration: ${iter + 1}/${maxIterations}`, 60, 30);
          ctx.fillText(`Inertia: ${kmeans.inertia.toFixed(2)}`, 60, 50);
        },
        onComplete: () => {
          // Actually update centroids after animation
          if (iter < maxIterations - 1) {
            kmeans.updateCentroids();
          }
        }
      });
    }
    
    // Phase 4: Final reveal with all elements
    timeline.add({
      duration: 2000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(data, kmeans.centroids, kmeans.assignments, 1);
        drawCentroids(kmeans.centroids, 1, false);
        
        if (params.show_assignments) {
          drawAssignmentLines(data, kmeans.centroids, kmeans.assignments, 1);
        }
        
        if (params.show_boundaries) {
          drawVoronoiBoundaries(kmeans.centroids, 1);
        }
        
        if (params.show_elbow) {
          // For elbow method, we need to run K-Means for different K values
          const inertiaValues = [];
          for (let k = 1; k <= 5; k++) {
            const testKmeans = new KMeans(k, data);
            testKmeans.run(10);
            inertiaValues.push(testKmeans.inertia);
          }
          drawElbowChart(inertiaValues, progress);
        }
        
        // Draw final info
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`K-Means Clustering (K=${params.n_clusters})`, 60, 30);
        ctx.fillText(`Final Inertia: ${kmeans.inertia.toFixed(2)}`, 60, 50);
        ctx.fillText(`Iterations: ${maxIterations}`, 60, 70);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'cluster-formation':
      params.show_assignments = true;
      params.show_boundaries = false;
      params.show_elbow = false;
      animateKMeans();
      break;
      
    case 'centroid-movement':
      params.show_assignments = false;
      params.show_boundaries = false;
      params.show_elbow = false;
      animateKMeans();
      break;
      
    case 'voronoi-diagram':
      params.show_assignments = false;
      params.show_boundaries = true;
      params.show_elbow = false;
      animateKMeans();
      break;
      
    case 'elbow-method':
      params.show_assignments = false;
      params.show_boundaries = false;
      params.show_elbow = true;
      animateKMeans();
      break;
      
    case 'all':
      params.show_assignments = true;
      params.show_boundaries = true;
      params.show_elbow = true;
      animateKMeans();
      break;
      
    default:
      animateKMeans();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Samples',
        type: 'range',
        min: 50,
        max: 500,
        step: 10,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizers['k-means'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Number of Clusters (K)',
        type: 'range',
        min: 1,
        max: 8,
        step: 1,
        value: params.n_clusters,
        onChange: (value) => {
          params.n_clusters = parseInt(value);
          visualizers['k-means'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Cluster Standard Deviation',
        type: 'range',
        min: 0.1,
        max: 2,
        step: 0.1,
        value: params.cluster_std,
        onChange: (value) => {
          params.cluster_std = parseFloat(value);
          visualizers['k-means'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Initialization Method',
        type: 'select',
        options: [
          { value: 'k-means++', label: 'K-Means++', selected: params.init_method === 'k-means++' },
          { value: 'random', label: 'Random', selected: params.init_method === 'random' }
        ],
        onChange: (value) => {
          params.init_method = value;
          visualizers['k-means'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Assignment Lines',
        type: 'checkbox',
        checked: params.show_assignments,
        onChange: (value) => {
          params.show_assignments = value;
          visualizers['k-means'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Voronoi Boundaries',
        type: 'checkbox',
        checked: params.show_boundaries,
        onChange: (value) => {
          params.show_boundaries = value;
          visualizers['k-means'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Elbow Method',
        type: 'checkbox',
        checked: params.show_elbow,
        onChange: (value) => {
          params.show_elbow = value;
          visualizers['k-means'](containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'cluster-formation', label: 'Cluster Formation', selected: visualizationType === 'cluster-formation' },
          { value: 'centroid-movement', label: 'Centroid Movement', selected: visualizationType === 'centroid-movement' },
          { value: 'voronoi-diagram', label: 'Voronoi Diagram', selected: visualizationType === 'voronoi-diagram' },
          { value: 'elbow-method', label: 'Elbow Method', selected: visualizationType === 'elbow-method' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizers['k-means'](containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'K-Means Parameters',
      description: 'Adjust parameters to see how they affect K-Means clustering.'
    });
  }
}

// =============================================
// Enhanced Hierarchical Clustering Visualizations
// =============================================
function visualizeHierarchicalClustering(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 100,
    n_clusters: 3,
    linkage: 'ward',
    distance_metric: 'euclidean',
    show_dendrogram: true,
    show_clusters: true,
    show_heatmap: false,
    show_cut_line: true,
    animation_duration: 3000,
    interactive: true,
    controlsContainer: null,
    distribution: 'blobs',
    show_grid: true,
    show_axes: true
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 1000;
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
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 350);
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
        ctx.lineTo(cinemaX, toCanvasY(bounds.yMax));
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
      ctx.fillText('Feature 1 (X)', width - 350 - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing with class-by-class animation
  const drawDataPoints = (points, progress = 1, clusters = null) => {
    ctx.save();
    
    points.forEach(point => {
      // Determine cluster color
      let color;
      if (clusters && clusters[point.id]) {
        color = COLORS.spectrum[clusters[point.id] % COLORS.spectrum.length];
      } else if (point.trueCluster !== undefined) {
        color = COLORS.spectrum[point.trueCluster % COLORS.spectrum.length];
      } else {
        color = COLORS.gray;
      }
      
      ctx.globalAlpha = progress;
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 6, 0, Math.PI * 2);
      
      ctx.fillStyle = color;
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    
    ctx.restore();
  };
  
  // Hierarchical clustering implementation
  class HierarchicalClustering {
    constructor(data, linkage = 'ward', distanceMetric = 'euclidean') {
      this.data = data.map((point, i) => ({ ...point, id: i }));
      this.linkage = linkage;
      this.distanceMetric = distanceMetric;
      this.clusters = this.data.map(point => [point]);
      this.distanceMatrix = this.calculateDistanceMatrix();
      this.mergeHistory = [];
      this.dendrogram = [];
    }
    
    calculateDistanceMatrix() {
      const n = this.data.length;
      const matrix = Array(n).fill().map(() => Array(n).fill(0));
      
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const dist = this.calculateDistance(this.data[i], this.data[j]);
          matrix[i][j] = dist;
          matrix[j][i] = dist;
        }
      }
      
      return matrix;
    }
    
    calculateDistance(a, b) {
      if (this.distanceMetric === 'euclidean') {
        return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
      } else if (this.distanceMetric === 'manhattan') {
        return Math.abs(a.x - b.x) + Math.abs(a.y - b.y);
      } else if (this.distanceMetric === 'cosine') {
        const dot = a.x * b.x + a.y * b.y;
        const magA = Math.sqrt(a.x * a.x + a.y * a.y);
        const magB = Math.sqrt(b.x * b.x + b.y * b.y);
        return 1 - (dot / (magA * magB));
      }
      return Math.sqrt(Math.pow(a.x - b.x, 2) + Math.pow(a.y - b.y, 2));
    }
    
    calculateClusterDistance(clusterA, clusterB) {
      if (this.linkage === 'single') {
        // Single linkage: minimum distance between points in clusters
        let minDist = Infinity;
        for (const pointA of clusterA) {
          for (const pointB of clusterB) {
            const dist = this.distanceMatrix[pointA.id][pointB.id];
            if (dist < minDist) minDist = dist;
          }
        }
        return minDist;
      } else if (this.linkage === 'complete') {
        // Complete linkage: maximum distance between points in clusters
        let maxDist = -Infinity;
        for (const pointA of clusterA) {
          for (const pointB of clusterB) {
            const dist = this.distanceMatrix[pointA.id][pointB.id];
            if (dist > maxDist) maxDist = dist;
          }
        }
        return maxDist;
      } else if (this.linkage === 'average') {
        // Average linkage: average distance between points in clusters
        let totalDist = 0;
        let count = 0;
        for (const pointA of clusterA) {
          for (const pointB of clusterB) {
            totalDist += this.distanceMatrix[pointA.id][pointB.id];
            count++;
          }
        }
        return totalDist / count;
      } else {
        // Ward's method: minimizes variance of merged clusters
        const centroidA = this.calculateCentroid(clusterA);
        const centroidB = this.calculateCentroid(clusterB);
        const dist = this.calculateDistance(centroidA, centroidB);
        return dist * Math.sqrt((clusterA.length * clusterB.length) / (clusterA.length + clusterB.length));
      }
    }
    
    calculateCentroid(cluster) {
      const sumX = cluster.reduce((sum, point) => sum + point.x, 0);
      const sumY = cluster.reduce((sum, point) => sum + point.y, 0);
      return {
        x: sumX / cluster.length,
        y: sumY / cluster.length
      };
    }
    
    findClosestClusters() {
      let minDistance = Infinity;
      let clusterAIndex = -1;
      let clusterBIndex = -1;
      
      for (let i = 0; i < this.clusters.length; i++) {
        for (let j = i + 1; j < this.clusters.length; j++) {
          const distance = this.calculateClusterDistance(this.clusters[i], this.clusters[j]);
          if (distance < minDistance) {
            minDistance = distance;
            clusterAIndex = i;
            clusterBIndex = j;
          }
        }
      }
      
      return { clusterAIndex, clusterBIndex, distance: minDistance };
    }
    
    mergeClusters() {
      if (this.clusters.length <= 1) return null;
      
      const { clusterAIndex, clusterBIndex, distance } = this.findClosestClusters();
      
      // Record the merge
      const mergeRecord = {
        clusterA: this.clusters[clusterAIndex],
        clusterB: this.clusters[clusterBIndex],
        distance: distance,
        newCluster: [...this.clusters[clusterAIndex], ...this.clusters[clusterBIndex]]
      };
      
      this.mergeHistory.push(mergeRecord);
      
      // Update dendrogram
      this.dendrogram.push({
        left: clusterAIndex,
        right: clusterBIndex,
        distance: distance,
        size: mergeRecord.newCluster.length
      });
      
      // Perform the merge
      this.clusters[clusterAIndex] = mergeRecord.newCluster;
      this.clusters.splice(clusterBIndex, 1);
      
      return mergeRecord;
    }
    
    cluster(nClusters = 1) {
      while (this.clusters.length > nClusters) {
        const mergeResult = this.mergeClusters();
        if (!mergeResult) break;
      }
      
      // Create cluster assignments
      const assignments = {};
      this.clusters.forEach((cluster, clusterIdx) => {
        cluster.forEach(point => {
          assignments[point.id] = clusterIdx;
        });
      });
      
      return assignments;
    }
  }
  
  // Enhanced dendrogram drawing
  const drawDendrogram = (hc, progress = 1, cutHeight = null) => {
    if (!params.show_dendrogram) return;
    
    ctx.save();
    
    // Dendrogram area
    const dendroX = width - 320;
    const dendroY = 50;
    const dendroWidth = 300;
    const dendroHeight = height - 100;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(dendroX, dendroY, dendroWidth, dendroHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(dendroX, dendroY, dendroWidth, dendroHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Dendrogram', dendroX + dendroWidth / 2, dendroY - 10);
    
    // Calculate scaling
    const maxDistance = Math.max(...hc.mergeHistory.map(m => m.distance));
    const scaleX = dendroWidth / (hc.data.length - 1);
    const scaleY = dendroHeight / (maxDistance * 1.1);
    
    // Draw cutoff line if enabled
    if (params.show_cut_line && cutHeight !== null) {
      const yPos = dendroY + dendroHeight - (cutHeight * scaleY);
      
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.beginPath();
      ctx.moveTo(dendroX, yPos);
      ctx.lineTo(dendroX + dendroWidth, yPos);
      ctx.stroke();
      ctx.setLineDash([]);
      
      // Draw cutoff label
      ctx.fillStyle = COLORS.accent;
      ctx.font = '12px Arial';
      ctx.textAlign = 'right';
      ctx.fillText(`Cut: ${cutHeight.toFixed(2)}`, dendroX - 5, yPos);
    }
    
    // Draw dendrogram
    const leafPositions = {};
    let leafCounter = 0;
    
    // Calculate leaf positions
    hc.data.forEach((point, i) => {
      leafPositions[i] = dendroX + leafCounter * scaleX;
      leafCounter += 1;
    });
    
    // Draw merge history
    const mergeProgress = Math.floor(hc.mergeHistory.length * progress);
    const visibleMerges = hc.mergeHistory.slice(0, mergeProgress);
    
    visibleMerges.forEach((merge, idx) => {
      const alpha = MathUtils.clamp((progress - idx / hc.mergeHistory.length) * hc.mergeHistory.length, 0, 1);
      ctx.globalAlpha = alpha;
      
      // Calculate positions for this merge
      const leftPos = leafPositions[merge.clusterA[0].id];
      const rightPos = leafPositions[merge.clusterB[0].id];
      const centerPos = (leftPos + rightPos) / 2;
      const yPos = dendroY + dendroHeight - (merge.distance * scaleY);
      
      // Draw horizontal line
      ctx.strokeStyle = COLORS.primary;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(leftPos, yPos);
      ctx.lineTo(rightPos, yPos);
      ctx.stroke();
      
      // Draw vertical lines
      ctx.beginPath();
      ctx.moveTo(leftPos, yPos);
      ctx.lineTo(leftPos, dendroY + dendroHeight);
      ctx.stroke();
      
      ctx.beginPath();
      ctx.moveTo(rightPos, yPos);
      ctx.lineTo(rightPos, dendroY + dendroHeight);
      ctx.stroke();
      
      // Update leaf position for parent cluster
      leafPositions[merge.newCluster[0].id] = centerPos;
      
      // Pulse effect for current merge
      if (idx === mergeProgress - 1 && progress < 1) {
        const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 200);
        ctx.strokeStyle = COLORS.highlight;
        ctx.lineWidth = 2 + 2 * pulse;
        ctx.beginPath();
        ctx.moveTo(leftPos, yPos);
        ctx.lineTo(rightPos, yPos);
        ctx.stroke();
      }
    });
    
    ctx.restore();
  };
  
  // Enhanced cluster visualization
  const drawClusters = (hc, progress = 1, cutHeight = null) => {
    if (!params.show_clusters) return;
    
    ctx.save();
    
    // Determine current clusters based on progress
    const mergeProgress = Math.floor(hc.mergeHistory.length * progress);
    const currentClusters = [];
    
    if (mergeProgress === 0) {
      // Initial state - all points are separate clusters
      hc.data.forEach(point => {
        currentClusters.push([point]);
      });
    } else {
      // Reconstruct clusters at this merge progress
      const tempClusters = hc.data.map(point => [point]);
      
      for (let i = 0; i < mergeProgress; i++) {
        const merge = hc.mergeHistory[i];
        // Find the clusters to merge
        let clusterAIndex = -1;
        let clusterBIndex = -1;
        
        for (let j = 0; j < tempClusters.length; j++) {
          if (tempClusters[j][0].id === merge.clusterA[0].id) {
            clusterAIndex = j;
          }
          if (tempClusters[j][0].id === merge.clusterB[0].id) {
            clusterBIndex = j;
          }
        }
        
        if (clusterAIndex !== -1 && clusterBIndex !== -1) {
          // Merge the clusters
          tempClusters[clusterAIndex] = [...tempClusters[clusterAIndex], ...tempClusters[clusterBIndex]];
          tempClusters.splice(clusterBIndex, 1);
        }
      }
      
      currentClusters.push(...tempClusters);
    }
    
    // Draw cluster centroids with animation
    currentClusters.forEach((cluster, clusterIdx) => {
      const centroid = {
        x: cluster.reduce((sum, p) => sum + p.x, 0) / cluster.length,
        y: cluster.reduce((sum, p) => sum + p.y, 0) / cluster.length
      };
      
      // Draw connecting lines for current cluster
      if (cluster.length > 1) {
        ctx.strokeStyle = COLORS.spectrum[clusterIdx % COLORS.spectrum.length] + '60';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        
        ctx.beginPath();
        ctx.moveTo(toCanvasX(centroid.x), toCanvasY(centroid.y));
        
        cluster.forEach(point => {
          ctx.lineTo(toCanvasX(point.x), toCanvasY(point.y));
        });
        
        ctx.stroke();
        ctx.setLineDash([]);
      }
      
      // Draw centroid with pulsing effect
      const pulse = 0.7 + 0.3 * Math.sin(Date.now() / 300);
      ctx.fillStyle = COLORS.spectrum[clusterIdx % COLORS.spectrum.length];
      ctx.beginPath();
      ctx.arc(toCanvasX(centroid.x), toCanvasY(centroid.y), 10 * pulse, 0, Math.PI * 2);
      ctx.fill();
      
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      
      // Draw cluster size
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(cluster.length.toString(), toCanvasX(centroid.x), toCanvasY(centroid.y) + 5);
    });
    
    // Draw data points
    drawDataPoints(hc.data, progress, null);
    
    ctx.restore();
  };
  
  // Enhanced heatmap drawing
  const drawHeatmap = (hc, progress = 1) => {
    if (!params.show_heatmap) return;
    
    ctx.save();
    
    // Heatmap area
    const heatmapSize = 200;
    const heatmapX = width - 320;
    const heatmapY = height - 270;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(heatmapX, heatmapY, heatmapSize, heatmapSize);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(heatmapX, heatmapY, heatmapSize, heatmapSize);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Distance Matrix', heatmapX + heatmapSize / 2, heatmapY - 10);
    
    // Draw heatmap cells
    const cellSize = heatmapSize / hc.data.length;
    
    for (let i = 0; i < hc.data.length; i++) {
      for (let j = 0; j < hc.data.length; j++) {
        const distance = hc.distanceMatrix[i][j];
        const maxDist = Math.max(...hc.distanceMatrix.flat());
        const intensity = distance / maxDist;
        
        // Use a color gradient from white to blue
        const colorValue = Math.floor(255 * intensity);
        ctx.fillStyle = `rgb(${colorValue}, ${colorValue}, 255)`;
        
        ctx.fillRect(
          heatmapX + i * cellSize,
          heatmapY + j * cellSize,
          cellSize,
          cellSize
        );
      }
    }
    
    // Draw dendrogram on top of heatmap if needed
    if (params.show_dendrogram) {
      // Simplified dendrogram drawing for heatmap
      const scale = heatmapSize / (hc.data.length - 1);
      const maxDistance = Math.max(...hc.mergeHistory.map(m => m.distance));
      const scaleY = heatmapSize / (maxDistance * 1.1);
      
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1;
      
      // Draw a simplified version of the dendrogram
      // This would be more complex in a real implementation
    }
    
    ctx.restore();
  };
  
  // Animate the hierarchical clustering with enhanced cinematic effects
  const animateHierarchicalClustering = () => {
    const hc = new HierarchicalClustering(data, params.linkage, params.distance_metric);
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing
    timeline.add({
      duration: 1500,
      easing: 'easeOutBack',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        
        // Draw points with animation
        const pointsToShow = Math.floor(data.length * progress);
        drawDataPoints(data.slice(0, pointsToShow), progress);
      }
    });
    
    // Phase 2: Animate distance matrix calculation
    timeline.add({
      duration: 1000,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 1);
        
        // Animate distance matrix visualization
        if (params.show_heatmap) {
          ctx.save();
          ctx.globalAlpha = progress;
          drawHeatmap(hc, progress);
          ctx.restore();
        }
        
        // Draw progress text
        ctx.fillStyle = COLORS.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Calculating distance matrix...', 60, 30);
      }
    });
    
    // Phase 3: Animate merging process
    const totalMerges = data.length - params.n_clusters;
    
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        
        // Perform merges up to current progress
        const targetMerges = Math.floor(progress * totalMerges);
        while (hc.mergeHistory.length < targetMerges && hc.clusters.length > params.n_clusters) {
          hc.mergeClusters();
        }
        
        // Draw current state
        drawClusters(hc, progress);
        drawDendrogram(hc, progress);
        
        if (params.show_heatmap) {
          drawHeatmap(hc, 1);
        }
        
        // Draw progress info
        ctx.fillStyle = COLORS.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Merges: ${hc.mergeHistory.length}/${totalMerges}`, 60, 30);
        ctx.fillText(`Clusters: ${hc.clusters.length}`, 60, 50);
        ctx.fillText(`Linkage: ${params.linkage}`, 60, 70);
      }
    });
    
    // Phase 4: Final reveal with cut line animation
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        
        // Calculate cut height based on progress
        const maxDistance = Math.max(...hc.mergeHistory.map(m => m.distance));
        const cutHeight = maxDistance * 0.7 * progress;
        
        // Draw final state
        const finalClusters = hc.cluster(params.n_clusters);
        drawDataPoints(data, 1, finalClusters);
        drawDendrogram(hc, 1, cutHeight);
        
        if (params.show_heatmap) {
          drawHeatmap(hc, 1);
        }
        
        // Draw final info
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Final Clusters: ${params.n_clusters}`, 60, 30);
        ctx.fillText(`Cut Height: ${cutHeight.toFixed(2)}`, 60, 50);
        ctx.fillText(`Linkage: ${params.linkage}`, 60, 70);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'dendrogram':
      params.show_clusters = false;
      params.show_heatmap = false;
      animateHierarchicalClustering();
      break;
      
    case 'cluster-formation':
      params.show_dendrogram = false;
      params.show_heatmap = false;
      animateHierarchicalClustering();
      break;
      
    case 'heatmap-dendrogram':
      params.show_clusters = false;
      params.show_heatmap = true;
      animateHierarchicalClustering();
      break;
      
    case 'interactive-cut':
      params.show_clusters = true;
      params.show_dendrogram = true;
      params.show_heatmap = false;
      params.show_cut_line = true;
      animateHierarchicalClustering();
      break;
      
    case 'all':
      params.show_clusters = true;
      params.show_dendrogram = true;
      params.show_heatmap = true;
      params.show_cut_line = true;
      animateHierarchicalClustering();
      break;
      
    default:
      animateHierarchicalClustering();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Samples',
        type: 'range',
        min: 20,
        max: 200,
        step: 10,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizeHierarchicalClustering(containerId, visualizationType, params);
        }
      },
      {
        label: 'Number of Clusters',
        type: 'range',
        min: 2,
        max: 10,
        step: 1,
        value: params.n_clusters,
        onChange: (value) => {
          params.n_clusters = parseInt(value);
          visualizeHierarchicalClustering(containerId, visualizationType, params);
        }
      },
      {
        label: 'Linkage Method',
        type: 'select',
        options: [
          { value: 'ward', label: 'Ward', selected: params.linkage === 'ward' },
          { value: 'complete', label: 'Complete', selected: params.linkage === 'complete' },
          { value: 'average', label: 'Average', selected: params.linkage === 'average' },
          { value: 'single', label: 'Single', selected: params.linkage === 'single' }
        ],
        onChange: (value) => {
          params.linkage = value;
          visualizeHierarchicalClustering(containerId, visualizationType, params);
        }
      },
      {
        label: 'Distance Metric',
        type: 'select',
        options: [
          { value: 'euclidean', label: 'Euclidean', selected: params.distance_metric === 'euclidean' },
          { value: 'manhattan', label: 'Manhattan', selected: params.distance_metric === 'manhattan' },
          { value: 'cosine', label: 'Cosine', selected: params.distance_metric === 'cosine' }
        ],
        onChange: (value) => {
          params.distance_metric = value;
          visualizeHierarchicalClustering(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Dendrogram',
        type: 'checkbox',
        checked: params.show_dendrogram,
        onChange: (value) => {
          params.show_dendrogram = value;
          visualizeHierarchicalClustering(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Clusters',
        type: 'checkbox',
        checked: params.show_clusters,
        onChange: (value) => {
          params.show_clusters = value;
          visualizeHierarchicalClustering(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Heatmap',
        type: 'checkbox',
        checked: params.show_heatmap,
        onChange: (value) => {
          params.show_heatmap = value;
          visualizeHierarchicalClustering(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Cut Line',
        type: 'checkbox',
        checked: params.show_cut_line,
        onChange: (value) => {
          params.show_cut_line = value;
          visualizeHierarchicalClustering(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'dendrogram', label: 'Dendrogram', selected: visualizationType === 'dendrogram' },
          { value: 'cluster-formation', label: 'Cluster Formation', selected: visualizationType === 'cluster-formation' },
          { value: 'heatmap-dendrogram', label: 'Heatmap + Dendrogram', selected: visualizationType === 'heatmap-dendrogram' },
          { value: 'interactive-cut', label: 'Interactive Cutting', selected: visualizationType === 'interactive-cut' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizeHierarchicalClustering(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'Hierarchical Clustering Parameters',
      description: 'Adjust parameters to see how they affect the hierarchical clustering process.'
    });
  }
}

// =============================================
// Enhanced Gaussian Mixture Models Visualizations
// =============================================
function visualizeGaussianMixtureModels(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 300,
    n_components: 3,
    covariance_type: 'full',
    max_iter: 100,
    tol: 0.001,
    random_state: 42,
    show_components: true,
    show_probability: true,
    show_em_steps: false,
    show_responsibilities: false,
    animation_duration: 3000,
    interactive: true,
    controlsContainer: null,
    distribution: 'blobs',
    show_grid: true,
    show_axes: true
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 1000;
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
    n_clusters: params.n_components,
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
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 350);
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
      ctx.fillText('Feature 1 (X)', width - 350 - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing with responsibility visualization
  const drawDataPoints = (points, progress = 1, responsibilities = null) => {
    ctx.save();
    
    points.forEach((point, i) => {
      let color;
      let alpha = progress;
      
      if (responsibilities) {
        // Calculate color based on responsibilities (soft assignment)
        let r = 0, g = 0, b = 0;
        
        for (let j = 0; j < responsibilities[i].length; j++) {
          const resp = responsibilities[i][j];
          const componentColor = COLORS.spectrum[j % COLORS.spectrum.length];
          
          // Parse the color
          const hex = componentColor.replace('#', '');
          r += parseInt(hex.substring(0, 2), 16) * resp;
          g += parseInt(hex.substring(2, 4), 16) * resp;
          b += parseInt(hex.substring(4, 6), 16) * resp;
        }
        
        color = `rgb(${Math.floor(r)}, ${Math.floor(g)}, ${Math.floor(b)})`;
        
        // Add glow effect for points with high responsibility
        const maxResp = Math.max(...responsibilities[i]);
        if (maxResp > 0.7) {
          ctx.shadowBlur = 10 * maxResp;
          ctx.shadowColor = color;
        }
      } else if (point.trueCluster !== undefined) {
        color = COLORS.spectrum[point.trueCluster % COLORS.spectrum.length];
      } else {
        color = COLORS.gray;
      }
      
      ctx.globalAlpha = alpha;
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 6, 0, Math.PI * 2);
      
      ctx.fillStyle = color;
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    
    ctx.restore();
  };
  
  // Gaussian Mixture Model implementation
  class GaussianMixtureModel {
    constructor(nComponents, covarianceType = 'full') {
      this.nComponents = nComponents;
      this.covarianceType = covarianceType;
      this.means = [];
      this.covariances = [];
      this.weights = Array(nComponents).fill(1 / nComponents);
      this.responsibilities = [];
      this.logLikelihood = 0;
      this.converged = false;
    }
    
    initialize(data) {
      // Initialize means with random data points
      const shuffled = [...data].sort(() => Math.random() - 0.5);
      this.means = shuffled.slice(0, this.nComponents).map(point => ({ x: point.x, y: point.y }));
      
      // Initialize covariances based on data variance
      const xVariance = MathUtils.variance(data.map(p => p.x));
      const yVariance = MathUtils.variance(data.map(p => p.y));
      
      for (let i = 0; i < this.nComponents; i++) {
        if (this.covarianceType === 'full') {
          this.covariances.push({
            xx: xVariance,
            xy: 0,
            yx: 0,
            yy: yVariance
          });
        } else if (this.covarianceType === 'diag') {
          this.covariances.push({
            xx: xVariance,
            yy: yVariance
          });
        } else { // spherical
          const avgVariance = (xVariance + yVariance) / 2;
          this.covariances.push({
            xx: avgVariance,
            yy: avgVariance
          });
        }
      }
    }
    
    multivariateGaussian(point, mean, covariance) {
      const dx = point.x - mean.x;
      const dy = point.y - mean.y;
      
      let det, invCov;
      
      if (this.covarianceType === 'full') {
        // For full covariance matrix
        det = covariance.xx * covariance.yy - covariance.xy * covariance.yx;
        if (det <= 0) det = 1e-6; // Avoid division by zero
        
        invCov = {
          xx: covariance.yy / det,
          xy: -covariance.xy / det,
          yx: -covariance.yx / det,
          yy: covariance.xx / det
        };
        
        const exponent = -0.5 * (
          dx * (invCov.xx * dx + invCov.xy * dy) +
          dy * (invCov.yx * dx + invCov.yy * dy)
        );
        
        return Math.exp(exponent) / (2 * Math.PI * Math.sqrt(det));
      } else {
        // For diagonal or spherical covariance
        det = covariance.xx * covariance.yy;
        if (det <= 0) det = 1e-6;
        
        const exponent = -0.5 * (
          (dx * dx) / covariance.xx +
          (dy * dy) / covariance.yy
        );
        
        return Math.exp(exponent) / (2 * Math.PI * Math.sqrt(det));
      }
    }
    
    expectation(data) {
      this.responsibilities = [];
      let newLogLikelihood = 0;
      
      for (const point of data) {
        const pointResponsibilities = [];
        let pointProbability = 0;
        
        for (let i = 0; i < this.nComponents; i++) {
          const componentProb = this.multivariateGaussian(point, this.means[i], this.covariances[i]);
          const weightedProb = this.weights[i] * componentProb;
          pointResponsibilities.push(weightedProb);
          pointProbability += weightedProb;
        }
        
        // Normalize responsibilities
        for (let i = 0; i < this.nComponents; i++) {
          pointResponsibilities[i] /= pointProbability;
        }
        
        this.responsibilities.push(pointResponsibilities);
        newLogLikelihood += Math.log(pointProbability);
      }
      
      // Check for convergence
      const logLikelihoodChange = Math.abs(newLogLikelihood - this.logLikelihood);
      this.logLikelihood = newLogLikelihood;
      
      if (logLikelihoodChange < 1e-3) {
        this.converged = true;
      }
      
      return this.responsibilities;
    }
    
    maximization(data) {
      const n = data.length;
      const totalResponsibilities = Array(this.nComponents).fill(0);
      
      // Calculate total responsibilities per component
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < this.nComponents; j++) {
          totalResponsibilities[j] += this.responsibilities[i][j];
        }
      }
      
      // Update weights
      for (let j = 0; j < this.nComponents; j++) {
        this.weights[j] = totalResponsibilities[j] / n;
      }
      
      // Update means
      for (let j = 0; j < this.nComponents; j++) {
        let sumX = 0;
        let sumY = 0;
        
        for (let i = 0; i < n; i++) {
          sumX += this.responsibilities[i][j] * data[i].x;
          sumY += this.responsibilities[i][j] * data[i].y;
        }
        
        this.means[j] = {
          x: sumX / totalResponsibilities[j],
          y: sumY / totalResponsibilities[j]
        };
      }
      
      // Update covariances
      for (let j = 0; j < this.nComponents; j++) {
        let sumXX = 0;
        let sumXY = 0;
        let sumYY = 0;
        
        for (let i = 0; i < n; i++) {
          const dx = data[i].x - this.means[j].x;
          const dy = data[i].y - this.means[j].y;
          
          sumXX += this.responsibilities[i][j] * dx * dx;
          sumXY += this.responsibilities[i][j] * dx * dy;
          sumYY += this.responsibilities[i][j] * dy * dy;
        }
        
        if (this.covarianceType === 'full') {
          this.covariances[j] = {
            xx: sumXX / totalResponsibilities[j],
            xy: sumXY / totalResponsibilities[j],
            yx: sumXY / totalResponsibilities[j],
            yy: sumYY / totalResponsibilities[j]
          };
        } else if (this.covarianceType === 'diag') {
          this.covariances[j] = {
            xx: sumXX / totalResponsibilities[j],
            yy: sumYY / totalResponsibilities[j]
          };
        } else { // spherical
          const avgVariance = (sumXX + sumYY) / (2 * totalResponsibilities[j]);
          this.covariances[j] = {
            xx: avgVariance,
            yy: avgVariance
          };
        }
        
        // Add small value to avoid singular matrices
        this.covariances[j].xx += 1e-6;
        this.covariances[j].yy += 1e-6;
      }
    }
    
    fit(data, maxIter = 100) {
      this.initialize(data);
      this.converged = false;
      
      for (let iter = 0; iter < maxIter && !this.converged; iter++) {
        this.expectation(data);
        this.maximization(data);
      }
      
      return this;
    }
  }
  
  // Enhanced Gaussian component visualization
  const drawGaussianComponents = (gmm, progress = 1) => {
    if (!params.show_components) return;
    
    ctx.save();
    
    gmm.means.forEach((mean, i) => {
      const covariance = gmm.covariances[i];
      const weight = gmm.weights[i];
      const color = COLORS.spectrum[i % COLORS.spectrum.length];
      
      // Draw ellipse representing the Gaussian
      ctx.globalAlpha = 0.3 * weight * progress;
      ctx.fillStyle = color;
      
      // Calculate ellipse parameters
      let width, height, angle;
      
      if (gmm.covarianceType === 'full') {
        // Calculate eigenvalues and eigenvectors for full covariance
        const a = covariance.xx;
        const b = covariance.xy;
        const c = covariance.yy;
        
        const trace = a + c;
        const det = a * c - b * b;
        
        const eigenvalue1 = (trace + Math.sqrt(trace * trace - 4 * det)) / 2;
        const eigenvalue2 = (trace - Math.sqrt(trace * trace - 4 * det)) / 2;
        
        width = Math.sqrt(eigenvalue1) * 2;
        height = Math.sqrt(eigenvalue2) * 2;
        
        // Calculate rotation angle
        angle = 0.5 * Math.atan2(2 * b, a - c);
      } else {
        width = Math.sqrt(covariance.xx) * 2;
        height = Math.sqrt(covariance.yy) * 2;
        angle = 0;
      }
      
      // Draw ellipse
      ctx.beginPath();
      ctx.ellipse(
        toCanvasX(mean.x),
        toCanvasY(mean.y),
        width * 50, // Scale for visibility
        height * 50,
        angle,
        0,
        Math.PI * 2
      );
      ctx.fill();
      
      // Draw outline
      ctx.globalAlpha = progress;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.stroke();
      
      // Draw mean point
      ctx.beginPath();
      ctx.arc(toCanvasX(mean.x), toCanvasY(mean.y), 8, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      
      // Draw weight label
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`w: ${weight.toFixed(2)}`, toCanvasX(mean.x), toCanvasY(mean.y) + 25);
    });
    
    ctx.restore();
  };
  
  // Enhanced probability surface visualization
  const drawProbabilitySurface = (gmm, progress = 1) => {
    if (!params.show_probability) return;
    
    ctx.save();
    
    // Create offscreen canvas for probability surface
    const surfaceCanvas = document.createElement('canvas');
    surfaceCanvas.width = width - 400;
    surfaceCanvas.height = height - 100;
    const surfaceCtx = surfaceCanvas.getContext('2d');
    
    // Draw probability surface
    const stepX = (bounds.xMax - bounds.xMin) / surfaceCanvas.width;
    const stepY = (bounds.yMax - bounds.yMin) / surfaceCanvas.height;
    
    for (let px = 0; px < surfaceCanvas.width; px++) {
      for (let py = 0; py < surfaceCanvas.height; py++) {
        const x = bounds.xMin + px * stepX;
        const y = bounds.yMin + py * stepY;
        
        let totalProb = 0;
        let r = 0, g = 0, b = 0;
        
        for (let i = 0; i < gmm.nComponents; i++) {
          const prob = gmm.multivariateGaussian({ x, y }, gmm.means[i], gmm.covariances[i]);
          const weightedProb = gmm.weights[i] * prob;
          totalProb += weightedProb;
          
          // Add color contribution from this component
          const color = COLORS.spectrum[i % COLORS.spectrum.length];
          const hex = color.replace('#', '');
          r += parseInt(hex.substring(0, 2), 16) * weightedProb;
          g += parseInt(hex.substring(2, 4), 16) * weightedProb;
          b += parseInt(hex.substring(4, 6), 16) * weightedProb;
        }
        
        // Normalize color
        if (totalProb > 0) {
          r = Math.floor(r / totalProb);
          g = Math.floor(g / totalProb);
          b = Math.floor(b / totalProb);
        }
        
        // Set alpha based on probability
        const alpha = Math.min(1, totalProb * 10) * progress;
        
        surfaceCtx.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`;
        surfaceCtx.fillRect(px, py, 1, 1);
      }
    }
    
    // Draw the surface canvas onto the main canvas
    ctx.globalAlpha = 0.6 * progress;
    ctx.drawImage(surfaceCanvas, 50, 50, surfaceCanvas.width, surfaceCanvas.height);
    
    ctx.restore();
  };
  
  // Enhanced EM steps visualization
  const drawEMSteps = (gmm, step, progress = 1) => {
    if (!params.show_em_steps) return;
    
    ctx.save();
    
    // Draw step information
    ctx.fillStyle = COLORS.text;
    ctx.font = '16px Arial';
    ctx.textAlign = 'left';
    
    if (step === 'expectation') {
      ctx.fillText('E-Step: Calculating Responsibilities', 60, 30);
      ctx.font = '14px Arial';
      ctx.fillText('Assigning probabilities to each point for each component', 60, 50);
    } else {
      ctx.fillText('M-Step: Updating Parameters', 60, 30);
      ctx.font = '14px Arial';
      ctx.fillText('Recalculating means, covariances, and weights', 60, 50);
    }
    
    // Draw progress bar for current step
    const barWidth = 300;
    const barHeight = 10;
    const barX = 60;
    const barY = 70;
    
    ctx.fillStyle = COLORS.grid;
    ctx.fillRect(barX, barY, barWidth, barHeight);
    
    ctx.fillStyle = COLORS.primary;
    ctx.fillRect(barX, barY, barWidth * progress, barHeight);
    
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(barX, barY, barWidth, barHeight);
    
    ctx.restore();
  };
  
  // Animate the Gaussian Mixture Model with enhanced cinematic effects
  const animateGaussianMixtureModel = () => {
    const gmm = new GaussianMixtureModel(params.n_components, params.covariance_type);
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing
    timeline.add({
      duration: 1500,
      easing: 'easeOutBack',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        
        // Draw points with animation
        const pointsToShow = Math.floor(data.length * progress);
        drawDataPoints(data.slice(0, pointsToShow), progress);
      }
    });
    
    // Phase 2: Initialize GMM with random parameters
    timeline.add({
      duration: 1000,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 1);
        
        // Initialize GMM if not done yet
        if (progress > 0.5 && gmm.means.length === 0) {
          gmm.initialize(data);
        }
        
        // Draw initial components
        if (gmm.means.length > 0) {
          drawGaussianComponents(gmm, progress);
        }
        
        // Draw initialization text
        ctx.fillStyle = COLORS.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Initializing Gaussian components...', 60, 30);
      }
    });
    
    // Phase 3: Animate EM algorithm
    const maxIter = Math.min(params.max_iter, 10); // Limit iterations for animation
    
    for (let iter = 0; iter < maxIter; iter++) {
      // E-Step
      timeline.add({
        duration: 1500,
        easing: 'easeInOutQuad',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Perform E-step if not done yet
          if (progress > 0.5 && (!gmm.responsibilities.length || iter > 0)) {
            gmm.expectation(data);
          }
          
          // Draw current state
          drawDataPoints(data, 1, gmm.responsibilities);
          drawGaussianComponents(gmm, 1);
          
          if (params.show_probability) {
            drawProbabilitySurface(gmm, 1);
          }
          
          drawEMSteps(gmm, 'expectation', progress);
          
          // Draw iteration info
          ctx.fillStyle = COLORS.text;
          ctx.font = '14px Arial';
          ctx.textAlign = 'left';
          ctx.fillText(`Iteration: ${iter + 1}/${maxIter}`, 60, 90);
          ctx.fillText(`Log-Likelihood: ${gmm.logLikelihood.toFixed(2)}`, 60, 110);
        }
      });
      
      // M-Step
      timeline.add({
        duration: 1500,
        easing: 'easeInOutQuad',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Perform M-step if not done yet
          if (progress > 0.5) {
            gmm.maximization(data);
          }
          
          // Draw current state
          drawDataPoints(data, 1, gmm.responsibilities);
          drawGaussianComponents(gmm, 1);
          
          if (params.show_probability) {
            drawProbabilitySurface(gmm, 1);
          }
          
          drawEMSteps(gmm, 'maximization', progress);
          
          // Draw iteration info
          ctx.fillStyle = COLORS.text;
          ctx.font = '14px Arial';
          ctx.textAlign = 'left';
          ctx.fillText(`Iteration: ${iter + 1}/${maxIter}`, 60, 90);
          ctx.fillText(`Log-Likelihood: ${gmm.logLikelihood.toFixed(2)}`, 60, 110);
        }
      });
    }
    
    // Phase 4: Final reveal
    timeline.add({
      duration: 2000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        
        // Draw final state
        drawDataPoints(data, 1, gmm.responsibilities);
        drawGaussianComponents(gmm, 1);
        
        if (params.show_probability) {
          drawProbabilitySurface(gmm, progress);
        }
        
        // Draw final info
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`GMM with ${params.n_components} Components`, 60, 30);
        ctx.font = '14px Arial';
        ctx.fillText(`Covariance Type: ${params.covariance_type}`, 60, 50);
        ctx.fillText(`Final Log-Likelihood: ${gmm.logLikelihood.toFixed(2)}`, 60, 70);
        
        // Draw component weights
        for (let i = 0; i < gmm.weights.length; i++) {
          ctx.fillStyle = COLORS.spectrum[i % COLORS.spectrum.length];
          ctx.fillText(`Component ${i + 1}: ${gmm.weights[i].toFixed(3)}`, 60, 90 + i * 20);
        }
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'probability-surface':
      params.show_components = false;
      params.show_em_steps = false;
      params.show_responsibilities = false;
      animateGaussianMixtureModel();
      break;
      
    case 'component-visualization':
      params.show_probability = false;
      params.show_em_steps = false;
      params.show_responsibilities = false;
      animateGaussianMixtureModel();
      break;
      
    case 'em-steps':
      params.show_probability = false;
      params.show_responsibilities = false;
      animateGaussianMixtureModel();
      break;
      
    case 'with-responsibilities':
      params.show_probability = false;
      params.show_em_steps = false;
      animateGaussianMixtureModel();
      break;
      
    case 'all':
      params.show_components = true;
      params.show_probability = true;
      params.show_em_steps = true;
      params.show_responsibilities = true;
      animateGaussianMixtureModel();
      break;
      
    default:
      animateGaussianMixtureModel();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Samples',
        type: 'range',
        min: 50,
        max: 500,
        step: 50,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizeGaussianMixtureModels(containerId, visualizationType, params);
        }
      },
      {
        label: 'Number of Components',
        type: 'range',
        min: 1,
        max: 8,
        step: 1,
        value: params.n_components,
        onChange: (value) => {
          params.n_components = parseInt(value);
          visualizeGaussianMixtureModels(containerId, visualizationType, params);
        }
      },
      {
        label: 'Covariance Type',
        type: 'select',
        options: [
          { value: 'full', label: 'Full', selected: params.covariance_type === 'full' },
          { value: 'diag', label: 'Diagonal', selected: params.covariance_type === 'diag' },
          { value: 'spherical', label: 'Spherical', selected: params.covariance_type === 'spherical' }
        ],
        onChange: (value) => {
          params.covariance_type = value;
          visualizeGaussianMixtureModels(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Components',
        type: 'checkbox',
        checked: params.show_components,
        onChange: (value) => {
          params.show_components = value;
          visualizeGaussianMixtureModels(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Probability Surface',
        type: 'checkbox',
        checked: params.show_probability,
        onChange: (value) => {
          params.show_probability = value;
          visualizeGaussianMixtureModels(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show EM Steps',
        type: 'checkbox',
        checked: params.show_em_steps,
        onChange: (value) => {
          params.show_em_steps = value;
          visualizeGaussianMixtureModels(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Responsibilities',
        type: 'checkbox',
        checked: params.show_responsibilities,
        onChange: (value) => {
          params.show_responsibilities = value;
          visualizeGaussianMixtureModels(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'probability-surface', label: 'Probability Surface', selected: visualizationType === 'probability-surface' },
          { value: 'component-visualization', label: 'Component Visualization', selected: visualizationType === 'component-visualization' },
          { value: 'em-steps', label: 'EM Algorithm Steps', selected: visualizationType === 'em-steps' },
          { value: 'with-responsibilities', label: 'With Responsibilities', selected: visualizationType === 'with-responsibilities' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizeGaussianMixtureModels(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'Gaussian Mixture Model Parameters',
      description: 'Adjust parameters to see how they affect the Gaussian Mixture Model and EM algorithm.'
    });
  }
}

// =============================================
// Principal Component Analysis Visualization
// =============================================
function visualizePCA(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 200,
    n_features: 10,
    n_components: 2,
    variance_explained: 0.95,
    show_original: false,
    show_transformed: true,
    show_vectors: true,
    show_variance: true,
    show_reconstruction: false,
    animation_duration: 3000,
    interactive: true,
    controlsContainer: null,
    show_grid: true,
    show_axes: true,
    use_3d: false
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
  const data = DataSimulator.generateHighDimData({
    n_samples: params.n_samples,
    n_features: params.n_features,
    n_informative: Math.min(3, params.n_features),
    noise: 0.1,
    cluster_std: 1.0,
    n_clusters: 3
  });
  
  // PCA implementation
  const performPCA = (data, nComponents) => {
    // Center the data
    const means = [];
    for (let j = 0; j < params.n_features; j++) {
      means[j] = data.reduce((sum, point) => sum + point.values[j], 0) / data.length;
    }
    
    const centeredData = data.map(point => ({
      values: point.values.map((val, j) => val - means[j]),
      cluster: point.cluster,
      label: point.label
    }));
    
    // Calculate covariance matrix
    const covMatrix = [];
    for (let i = 0; i < params.n_features; i++) {
      covMatrix[i] = [];
      for (let j = 0; j < params.n_features; j++) {
        covMatrix[i][j] = centeredData.reduce((sum, point) => 
          sum + point.values[i] * point.values[j], 0) / (data.length - 1);
      }
    }
    
    // Simple eigenvalue decomposition (for visualization purposes)
    // In a real implementation, you'd use a proper eigen decomposition
    const eigenvalues = [];
    const eigenvectors = [];
    
    // For visualization, we'll create some meaningful eigenvalues/vectors
    const totalVariance = covMatrix.reduce((sum, row, i) => sum + row[i], 0);
    for (let i = 0; i < params.n_features; i++) {
      eigenvalues[i] = totalVariance * Math.pow(0.7, i); // Exponential decay
      eigenvectors[i] = Array(params.n_features).fill(0).map((_, j) => 
        MathUtils.random(-1, 1)
      );
      
      // Normalize eigenvector
      const mag = Math.sqrt(eigenvectors[i].reduce((sum, val) => sum + val * val, 0));
      eigenvectors[i] = eigenvectors[i].map(val => val / mag);
    }
    
    // Sort by eigenvalue (descending)
    const sortedIndices = eigenvalues.map((val, idx) => ({val, idx}))
      .sort((a, b) => b.val - a.val)
      .map(obj => obj.idx);
    
    const sortedEigenvalues = sortedIndices.map(i => eigenvalues[i]);
    const sortedEigenvectors = sortedIndices.map(i => eigenvectors[i]);
    
    // Project data onto principal components
    const projectedData = centeredData.map(point => {
      const projection = [];
      for (let i = 0; i < nComponents; i++) {
        projection[i] = MathUtils.dot(point.values, sortedEigenvectors[i]);
      }
      return {
        x: projection[0],
        y: projection[1],
        z: projection[2],
        cluster: point.cluster,
        label: point.label,
        original: point.values
      };
    });
    
    return {
      projectedData,
      eigenvalues: sortedEigenvalues,
      eigenvectors: sortedEigenvectors,
      means
    };
  };
  
  // Perform PCA
  const pcaResult = performPCA(data, params.n_components);
  const { projectedData, eigenvalues, eigenvectors } = pcaResult;
  
  // Find data bounds for scaling
  const xValues = projectedData.map(p => p.x);
  const yValues = projectedData.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const padding = 0.5;
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
      ctx.fillText('PC1', width - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('PC2', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing
  const drawDataPoints = (points, progress = 1) => {
    ctx.save();
    
    points.forEach(point => {
      ctx.globalAlpha = progress;
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 6, 0, Math.PI * 2);
      
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
  
  // Draw eigenvectors as arrows
  const drawEigenvectors = (progress = 1) => {
    if (!params.show_vectors) return;
    
    ctx.save();
    
    // Scale factor for eigenvectors
    const scale = 2;
    
    // Draw each eigenvector
    for (let i = 0; i < Math.min(2, eigenvectors.length); i++) {
      const magnitude = Math.sqrt(eigenvalues[i]) * scale;
      const dx = eigenvectors[i][0] * magnitude;
      const dy = eigenvectors[i][1] * magnitude;
      
      // Arrow color
      ctx.strokeStyle = COLORS.spectrum[i % COLORS.spectrum.length];
      ctx.fillStyle = COLORS.spectrum[i % COLORS.spectrum.length];
      ctx.lineWidth = 2;
      ctx.globalAlpha = progress;
      
      // Draw arrow line
      ctx.beginPath();
      ctx.moveTo(toCanvasX(0), toCanvasY(0));
      ctx.lineTo(toCanvasX(dx), toCanvasY(dy));
      ctx.stroke();
      
      // Draw arrowhead
      const angle = Math.atan2(dy, dx);
      ctx.beginPath();
      ctx.moveTo(toCanvasX(dx), toCanvasY(dy));
      ctx.lineTo(
        toCanvasX(dx - 10 * Math.cos(angle - Math.PI / 6)),
        toCanvasY(dy - 10 * Math.sin(angle - Math.PI / 6))
      );
      ctx.lineTo(
        toCanvasX(dx - 10 * Math.cos(angle + Math.PI / 6)),
        toCanvasY(dy - 10 * Math.sin(angle + Math.PI / 6))
      );
      ctx.closePath();
      ctx.fill();
      
      // Label the eigenvector
      ctx.fillStyle = COLORS.text;
      ctx.font = '14px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`PC${i + 1}`, toCanvasX(dx / 2), toCanvasY(dy / 2) - 10);
      
      // Add variance explained
      const variance = eigenvalues[i] / eigenvalues.reduce((a, b) => a + b, 0);
      ctx.fillText(`${(variance * 100).toFixed(1)}%`, toCanvasX(dx / 2), toCanvasY(dy / 2) + 15);
    }
    
    ctx.restore();
  };
  
  // Draw variance explained bar chart
  const drawVarianceChart = (progress = 1) => {
    if (!params.show_variance) return;
    
    ctx.save();
    
    // Chart dimensions
    const chartWidth = 300;
    const chartHeight = 200;
    const chartX = width - chartWidth - 20;
    const chartY = height - chartHeight - 20;
    
    // Draw chart background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(chartX, chartY, chartWidth, chartHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(chartX, chartY, chartWidth, chartHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Variance Explained', chartX + chartWidth / 2, chartY - 10);
    
    // Calculate total variance
    const totalVariance = eigenvalues.reduce((a, b) => a + b, 0);
    
    // Draw bars for each component
    const barWidth = chartWidth / Math.min(5, eigenvalues.length);
    const maxBarHeight = chartHeight * 0.8;
    
    for (let i = 0; i < Math.min(5, eigenvalues.length); i++) {
      const variance = eigenvalues[i] / totalVariance;
      const barHeight = variance * maxBarHeight * progress;
      const barX = chartX + i * barWidth + barWidth * 0.1;
      const barY = chartY + chartHeight - barHeight;
      
      // Draw bar
      ctx.fillStyle = COLORS.spectrum[i % COLORS.spectrum.length];
      ctx.fillRect(barX, barY, barWidth * 0.8, barHeight);
      
      // Draw label
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`PC${i + 1}`, barX + barWidth * 0.4, chartY + chartHeight + 15);
      
      // Draw variance percentage
      ctx.fillText(`${(variance * 100).toFixed(1)}%`, barX + barWidth * 0.4, barY - 5);
    }
    
    // Draw cumulative variance line
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    
    let cumulative = 0;
    for (let i = 0; i < Math.min(5, eigenvalues.length); i++) {
      cumulative += eigenvalues[i] / totalVariance;
      const lineX = chartX + (i + 0.5) * barWidth;
      const lineY = chartY + chartHeight - cumulative * maxBarHeight * progress;
      
      if (i === 0) {
        ctx.moveTo(lineX, lineY);
      } else {
        ctx.lineTo(lineX, lineY);
      }
    }
    
    ctx.stroke();
    ctx.setLineDash([]);
    
    // Draw cumulative labels
    cumulative = 0;
    for (let i = 0; i < Math.min(5, eigenvalues.length); i++) {
      cumulative += eigenvalues[i] / totalVariance;
      const lineX = chartX + (i + 0.5) * barWidth;
      const lineY = chartY + chartHeight - cumulative * maxBarHeight * progress;
      
      ctx.fillStyle = COLORS.accent;
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(`${(cumulative * 100).toFixed(1)}%`, lineX, lineY - 10);
    }
    
    ctx.restore();
  };
  
  // Draw scree plot
  const drawScreePlot = (progress = 1) => {
    ctx.save();
    
    // Plot dimensions
    const plotWidth = 300;
    const plotHeight = 200;
    const plotX = 20;
    const plotY = height - plotHeight - 20;
    
    // Draw plot background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(plotX, plotY, plotWidth, plotHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(plotX, plotY, plotWidth, plotHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Scree Plot', plotX + plotWidth / 2, plotY - 10);
    
    // Calculate total variance
    const totalVariance = eigenvalues.reduce((a, b) => a + b, 0);
    
    // Draw scree plot
    const maxEigenvalue = Math.max(...eigenvalues);
    const pointSpacing = plotWidth / (eigenvalues.length + 1);
    
    ctx.strokeStyle = COLORS.primary;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 0; i < eigenvalues.length; i++) {
      const pointX = plotX + (i + 1) * pointSpacing;
      const pointY = plotY + plotHeight - (eigenvalues[i] / maxEigenvalue) * plotHeight * 0.8 * progress;
      
      if (i === 0) {
        ctx.moveTo(pointX, pointY);
      } else {
        ctx.lineTo(pointX, pointY);
      }
      
      // Draw point
      ctx.fillStyle = COLORS.primary;
      ctx.beginPath();
      ctx.arc(pointX, pointY, 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw eigenvalue label
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(eigenvalues[i].toFixed(2), pointX, pointY - 10);
    }
    
    ctx.stroke();
    
    // Draw elbow point (simplified - just highlight component 2)
    if (eigenvalues.length >= 2 && progress > 0.8) {
      const elbowX = plotX + 2 * pointSpacing;
      const elbowY = plotY + plotHeight - (eigenvalues[1] / maxEigenvalue) * plotHeight * 0.8;
      
      ctx.fillStyle = COLORS.accent;
      ctx.beginPath();
      ctx.arc(elbowX, elbowY, 8, 0, Math.PI * 2);
      ctx.fill();
      
      // Pulse effect
      const pulseSize = 5 + 5 * Math.sin(Date.now() / 300);
      ctx.strokeStyle = COLORS.accent;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(elbowX, elbowY, 8 + pulseSize, 0, Math.PI * 2);
      ctx.stroke();
      
      // Label elbow point
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Optimal Components', elbowX, elbowY - 20);
    }
    
    ctx.restore();
  };
  
  // Animate the PCA process with enhanced cinematic effects
  const animatePCAProcess = () => {
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Show original high-dimensional data (conceptual)
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw conceptual high-dimensional data representation
        ctx.save();
        ctx.fillStyle = COLORS.text;
        ctx.font = '20px Arial';
        ctx.textAlign = 'center';
        ctx.globalAlpha = progress;
        ctx.fillText('High-Dimensional Data Space', width / 2, height / 2);
        
        // Draw points floating in 3D space (conceptual)
        for (let i = 0; i < 20; i++) {
          const x = MathUtils.random(100, width - 100);
          const y = MathUtils.random(100, height - 100);
          const size = MathUtils.random(3, 8);
          
          ctx.beginPath();
          ctx.arc(x, y, size, 0, Math.PI * 2);
          ctx.fillStyle = COLORS.spectrum[i % COLORS.spectrum.length];
          ctx.fill();
        }
        
        ctx.restore();
      }
    });
    
    // Phase 2: Show variance explained chart growing
    timeline.add({
      duration: 2000,
      easing: 'easeOutBounce',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawVarianceChart(progress);
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Variance Explained by Principal Components', width / 2, 30);
      }
    });
    
    // Phase 3: Show scree plot with elbow point
    timeline.add({
      duration: 2000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawScreePlot(progress);
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Scree Plot - Finding the Optimal Number of Components', width / 2, 30);
      }
    });
    
    // Phase 4: Show data projected with eigenvectors
    timeline.add({
      duration: 2500,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(projectedData, progress);
        drawEigenvectors(progress);
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Data Projected onto Principal Components', width / 2, 30);
      }
    });
    
    // Phase 5: Final view with all elements
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(projectedData, 1);
        drawEigenvectors(1);
        drawVarianceChart(1);
        drawScreePlot(1);
        
        // Draw final stats
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Total Variance Explained: ${(eigenvalues.slice(0, params.n_components).reduce((a, b) => a + b, 0) / eigenvalues.reduce((a, b) => a + b, 0) * 100).toFixed(1)}%`, 60, 90);
        ctx.fillText(`Number of Components: ${params.n_components}`, 60, 110);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'variance-explained':
      params.show_vectors = false;
      animatePCAProcess();
      break;
      
    case 'biplot':
      params.show_variance = false;
      animatePCAProcess();
      break;
      
    case 'scree-plot':
      params.show_vectors = false;
      params.show_variance = false;
      animatePCAProcess();
      break;
      
    case 'reconstruction':
      // Simplified reconstruction view
      animatePCAProcess();
      break;
      
    case 'all':
      animatePCAProcess();
      break;
      
    default:
      animatePCAProcess();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Samples',
        type: 'range',
        min: 50,
        max: 500,
        step: 10,
        value: params.n_samples,
        onChange: (value) => {
          params.n_samples = parseInt(value);
          visualizePCA(containerId, visualizationType, params);
        }
      },
      {
        label: 'Number of Components',
        type: 'range',
        min: 2,
        max: 5,
        step: 1,
        value: params.n_components,
        onChange: (value) => {
          params.n_components = parseInt(value);
          visualizePCA(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Eigenvectors',
        type: 'checkbox',
        checked: params.show_vectors,
        onChange: (value) => {
          params.show_vectors = value;
          visualizePCA(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Variance Chart',
        type: 'checkbox',
        checked: params.show_variance,
        onChange: (value) => {
          params.show_variance = value;
          visualizePCA(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'variance-explained', label: 'Variance Explained', selected: visualizationType === 'variance-explained' },
          { value: 'biplot', label: 'Biplot', selected: visualizationType === 'biplot' },
          { value: 'scree-plot', label: 'Scree Plot', selected: visualizationType === 'scree-plot' },
          { value: 'reconstruction', label: 'Reconstruction', selected: visualizationType === 'reconstruction' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizePCA(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'PCA Parameters',
      description: 'Adjust parameters to see how they affect the PCA transformation.'
    });
  }
}

// =============================================
// DBSCAN Visualization
// =============================================
function visualizeDBSCAN(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 150,
    n_clusters: 3,
    cluster_std: 0.5,
    epsilon: 0.3,
    min_samples: 5,
    distribution: 'blobs',
    noise_level: 0.05,
    show_epsilon: false,
    show_core_points: false,
    show_step_by_step: false,
    animation_duration: 3000,
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
  
  // Generate clustering data
  const data = DataSimulator.generateClusteringData({
    n_samples: params.n_samples,
    n_clusters: params.n_clusters,
    cluster_std: params.cluster_std,
    distribution: params.distribution,
    noise_level: params.noise_level
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
  const drawDataPoints = (points, progress = 1, highlight = null) => {
    ctx.save();
    
    points.forEach(point => {
      // Determine point properties based on cluster status
      let color, radius, alpha;
      
      if (point.cluster === -1) {
        // Noise point
        color = COLORS.gray;
        radius = 4;
        alpha = 0.6;
      } else {
        // Cluster point
        const colors = COLORS.spectrum;
        color = colors[point.cluster % colors.length];
        radius = 6;
        alpha = 1;
      }
      
      // Highlight specific points
      if (highlight && highlight.includes(point)) {
        radius += 2;
        alpha = 1;
      }
      
      ctx.globalAlpha = alpha * progress;
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), radius, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    
    ctx.restore();
  };
  
  // Draw epsilon circles around points
  const drawEpsilonCircles = (points, progress = 1) => {
    if (!params.show_epsilon) return;
    
    ctx.save();
    
    points.forEach(point => {
      const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 500 + point.x * 10);
      ctx.globalAlpha = 0.2 * progress * pulse;
      ctx.strokeStyle = point.cluster === -1 ? COLORS.gray : COLORS.spectrum[point.cluster % COLORS.spectrum.length];
      ctx.lineWidth = 1 + pulse;
      
      ctx.beginPath();
      ctx.arc(
        toCanvasX(point.x),
        toCanvasY(point.y),
        toCanvasX(point.x + params.epsilon) - toCanvasX(point.x),
        0,
        Math.PI * 2
      );
      ctx.stroke();
    });
    
    ctx.restore();
  };
  
  // Highlight core points
  const drawCorePoints = (points, progress = 1) => {
    if (!params.show_core_points) return;
    
    ctx.save();
    
    // Simple core point detection (for visualization)
    const corePoints = points.filter(point => {
      // Count neighbors within epsilon
      const neighbors = points.filter(p => 
        MathUtils.distance(point.x, point.y, p.x, p.y) <= params.epsilon
      );
      return neighbors.length >= params.min_samples;
    });
    
    corePoints.forEach(point => {
      // Pulse glow effect
      const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 300);
      ctx.globalAlpha = 0.7 * progress;
      
      // Outer glow
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 12 + pulse * 3, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.highlight + '40';
      ctx.fill();
      
      // Inner circle
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 8, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.highlight;
      ctx.fill();
      
      // Label
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText('Core', toCanvasX(point.x), toCanvasY(point.y) - 15);
    });
    
    ctx.restore();
  };
  
  // DBSCAN algorithm implementation (simplified for visualization)
  const performDBSCAN = (points) => {
    let clusterId = 0;
    const clusters = [];
    
    // Mark all points as unvisited
    points.forEach(point => {
      point.visited = false;
      point.cluster = -1; // -1 means noise
    });
    
    // Process each point
    points.forEach(point => {
      if (point.visited) return;
      
      point.visited = true;
      
      // Find neighbors
      const neighbors = points.filter(p => 
        MathUtils.distance(point.x, point.y, p.x, p.y) <= params.epsilon
      );
      
      if (neighbors.length < params.min_samples) {
        // Mark as noise
        point.cluster = -1;
      } else {
        // Start new cluster
        clusterId++;
        point.cluster = clusterId;
        clusters.push([point]);
        
        // Process neighbors
        let i = 0;
        while (i < neighbors.length) {
          const neighbor = neighbors[i];
          
          if (!neighbor.visited) {
            neighbor.visited = true;
            
            // Find neighbor's neighbors
            const neighborNeighbors = points.filter(p => 
              MathUtils.distance(neighbor.x, neighbor.y, p.x, p.y) <= params.epsilon
            );
            
            if (neighborNeighbors.length >= params.min_samples) {
              // Add to neighbors list
              neighbors.push(...neighborNeighbors.filter(n => !neighbors.includes(n)));
            }
          }
          
          // Add to cluster if not already in one
          if (neighbor.cluster === -1) {
            neighbor.cluster = clusterId;
            clusters[clusterId - 1].push(neighbor);
          }
          
          i++;
        }
      }
    });
    
    return clusters;
  };
  
  // Perform DBSCAN
  const clusters = performDBSCAN(data);
  
  // Animate the DBSCAN process with enhanced cinematic effects
  const animateDBSCANProcess = () => {
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Show raw data points
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, progress);
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Raw Data Points', width / 2, 30);
      }
    });
    
    // Phase 2: Show epsilon circles
    timeline.add({
      duration: 2000,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 1);
        drawEpsilonCircles(data, progress);
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Epsilon Neighborhoods (ε = ' + params.epsilon + ')', width / 2, 30);
      }
    });
    
    // Phase 3: Highlight core points
    timeline.add({
      duration: 2000,
      easing: 'easeOutBounce',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 1);
        drawCorePoints(data, progress);
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Identifying Core Points (MinPts = ' + params.min_samples + ')', width / 2, 30);
      }
    });
    
    // Phase 4: Show cluster formation
    timeline.add({
      duration: 2500,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        
        // Draw clusters forming one by one
        clusters.forEach((cluster, idx) => {
          const clusterProgress = Math.min(1, progress * (idx + 1) / clusters.length);
          drawDataPoints(cluster, clusterProgress);
        });
        
        // Draw noise points
        const noise = data.filter(p => p.cluster === -1);
        drawDataPoints(noise, progress);
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Cluster Formation', width / 2, 30);
      }
    });
    
    // Phase 5: Final view with all elements
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(data, 1);
        drawEpsilonCircles(data, 1);
        drawCorePoints(data, 1);
        
        // Draw final stats
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Clusters Found: ${clusters.length}`, 60, 90);
        ctx.fillText(`Noise Points: ${data.filter(p => p.cluster === -1).length}`, 60, 110);
        ctx.fillText(`Epsilon (ε): ${params.epsilon}`, 60, 130);
        ctx.fillText(`MinPts: ${params.min_samples}`, 60, 150);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'default':
      params.show_epsilon = false;
      params.show_core_points = false;
      animateDBSCANProcess();
      break;
      
    case 'step-by-step':
      params.show_step_by_step = true;
      animateDBSCANProcess();
      break;
      
    case 'epsilon-circles':
      params.show_core_points = false;
      animateDBSCANProcess();
      break;
      
    case 'core-points':
      params.show_epsilon = false;
      animateDBSCANProcess();
      break;
      
    case 'all':
      animateDBSCANProcess();
      break;
      
    default:
      animateDBSCANProcess();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Epsilon (ε)',
        type: 'range',
        min: 0.1,
        max: 1.0,
        step: 0.05,
        value: params.epsilon,
        onChange: (value) => {
          params.epsilon = parseFloat(value);
          visualizeDBSCAN(containerId, visualizationType, params);
        }
      },
      {
        label: 'MinPts',
        type: 'range',
        min: 2,
        max: 10,
        step: 1,
        value: params.min_samples,
        onChange: (value) => {
          params.min_samples = parseInt(value);
          visualizeDBSCAN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Epsilon Circles',
        type: 'checkbox',
        checked: params.show_epsilon,
        onChange: (value) => {
          params.show_epsilon = value;
          visualizeDBSCAN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Core Points',
        type: 'checkbox',
        checked: params.show_core_points,
        onChange: (value) => {
          params.show_core_points = value;
          visualizeDBSCAN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'default', label: 'Standard View', selected: visualizationType === 'default' },
          { value: 'step-by-step', label: 'Step-by-Step', selected: visualizationType === 'step-by-step' },
          { value: 'epsilon-circles', label: 'Epsilon Neighborhoods', selected: visualizationType === 'epsilon-circles' },
          { value: 'core-points', label: 'Core Points Highlight', selected: visualizationType === 'core-points' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizeDBSCAN(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'DBSCAN Parameters',
      description: 'Adjust parameters to see how they affect the clustering results.'
    });
  }
}

// =============================================
// Bagging Visualization
// =============================================
function visualizeBagging(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_samples: 100,
    n_estimators: 10,
    base_learner: 'decision-tree',
    max_depth: 3,
    bootstrap_ratio: 0.8,
    noise: 0.3,
    show_bootstraps: true,
    show_individuals: false,
    show_ensemble: true,
    show_variance: false,
    animation_duration: 3000,
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
  
  // Generate classification data
  const data = DataSimulator.generateClassificationData({
    n_samples: params.n_samples,
    n_features: 2,
    n_classes: 2,
    n_clusters: 2,
    noise: params.noise,
    random_state: 42
  });
  
  // Find data bounds for scaling
  const xValues = data.map(p => p.x);
  const yValues = data.map(p => p.y);
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  
  // Add padding to bounds
  const padding = 0.5;
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
  const drawDataPoints = (points, progress = 1, highlight = false) => {
    ctx.save();
    
    points.forEach(point => {
      // Different colors for different classes
      const colors = [COLORS.primary, COLORS.secondary];
      ctx.globalAlpha = highlight ? 1 : 0.7;
      ctx.globalAlpha *= progress;
      
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 6, 0, Math.PI * 2);
      ctx.fillStyle = colors[point.label % colors.length];
      ctx.fill();
      
      // Add outline for better visibility
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 0.5;
      ctx.stroke();
    });
    
    ctx.restore();
  };
  
  // Draw decision boundary (simplified for visualization)
  const drawDecisionBoundary = (progress = 1, color = COLORS.accent, widthMultiplier = 2) => {
    ctx.save();
    
    ctx.strokeStyle = color;
    ctx.lineWidth = widthMultiplier;
    ctx.globalAlpha = 0.8 * progress;
    
    // Simplified decision boundary (sine wave for visualization)
    ctx.beginPath();
    
    for (let x = bounds.xMin; x <= bounds.xMax; x += 0.05) {
      const y = Math.sin(x * 2) * 0.5 + 0.2;
      
      if (x === bounds.xMin) {
        ctx.moveTo(toCanvasX(x), toCanvasY(y));
      } else {
        ctx.lineTo(toCanvasX(x), toCanvasY(y));
      }
    }
    
    ctx.stroke();
    
    ctx.restore();
  };
  
  // Draw multiple decision boundaries for individual models
  const drawIndividualModels = (nModels, progress = 1) => {
    ctx.save();
    
    for (let i = 0; i < nModels; i++) {
      const alpha = 0.2 + 0.8 * (i / nModels);
      const offset = Math.sin(i * 0.5) * 0.3;
      const color = i % 2 === 0 ? COLORS.primary : COLORS.secondary;
      
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.3 * progress;
      
      // Draw slightly different decision boundary for each model
      ctx.beginPath();
      
      for (let x = bounds.xMin; x <= bounds.xMax; x += 0.05) {
        const y = Math.sin(x * 2 + offset) * 0.5 + 0.2;
        
        if (x === bounds.xMin) {
          ctx.moveTo(toCanvasX(x), toCanvasY(y));
        } else {
          ctx.lineTo(toCanvasX(x), toCanvasY(y));
        }
      }
      
      ctx.stroke();
    }
    
    ctx.restore();
  };
  
  // Draw bootstrap sampling visualization
  const drawBootstrapSamples = (progress = 1) => {
    if (!params.show_bootstraps) return;
    
    ctx.save();
    
    // Draw original dataset
    drawDataPoints(data, progress * 0.7);
    
    // Highlight bootstrap samples
    const sampleSize = Math.floor(data.length * params.bootstrap_ratio);
    const sampleIndices = Array(sampleSize).fill().map(() => 
      Math.floor(Math.random() * data.length)
    );
    const uniqueIndices = [...new Set(sampleIndices)];
    
    // Draw highlighted points
    uniqueIndices.forEach(idx => {
      const point = data[idx];
      
      // Pulse glow effect
      const pulse = 0.5 + 0.5 * Math.sin(Date.now() / 500 + idx * 0.1);
      ctx.globalAlpha = 0.8 * progress;
      
      // Outer glow
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 10 + pulse * 2, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.highlight + '40';
      ctx.fill();
      
      // Inner circle
      ctx.beginPath();
      ctx.arc(toCanvasX(point.x), toCanvasY(point.y), 6, 0, Math.PI * 2);
      ctx.fillStyle = COLORS.highlight;
      ctx.fill();
    });
    
    // Draw sample count
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'right';
    ctx.fillText(`Bootstrap Sample: ${uniqueIndices.length} unique points`, width - 20, 60);
    
    ctx.restore();
  };
  
  // Draw variance reduction chart
  const drawVarianceChart = (progress = 1) => {
    if (!params.show_variance) return;
    
    ctx.save();
    
    // Chart dimensions
    const chartWidth = 300;
    const chartHeight = 150;
    const chartX = width - chartWidth - 20;
    const chartY = height - chartHeight - 20;
    
    // Draw chart background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(chartX, chartY, chartWidth, chartHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(chartX, chartY, chartWidth, chartHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Variance Reduction', chartX + chartWidth / 2, chartY - 10);
    
    // Draw variance curve
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    for (let i = 1; i <= 10; i++) {
      const x = chartX + (i / 10) * chartWidth;
      const variance = 1 / Math.sqrt(i); // Simulated variance reduction
      const y = chartY + chartHeight - variance * chartHeight * 0.8 * progress;
      
      if (i === 1) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
      
      // Draw point
      ctx.fillStyle = COLORS.accent;
      ctx.beginPath();
      ctx.arc(x, y, 3, 0, Math.PI * 2);
      ctx.fill();
    }
    
    ctx.stroke();
    
    // Draw axis labels
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Number of Estimators', chartX + chartWidth / 2, chartY + chartHeight + 20);
    
    ctx.textAlign = 'right';
    ctx.fillText('Variance', chartX - 5, chartY + chartHeight / 2);
    
    ctx.restore();
  };
  
  // Animate the bagging process with enhanced cinematic effects
  const animateBaggingProcess = () => {
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Show original dataset
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, progress);
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Original Dataset', width / 2, 30);
      }
    });
    
    // Phase 2: Show bootstrap sampling
    timeline.add({
      duration: 2000,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawBootstrapSamples(progress);
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Bootstrap Sampling (Bagging)', width / 2, 30);
      }
    });
    
    // Phase 3: Show individual models
    timeline.add({
      duration: 2000,
      easing: 'easeOutBounce',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 0.7);
        drawIndividualModels(params.n_estimators, progress);
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Individual Models (High Variance)', width / 2, 30);
      }
    });
    
    // Phase 4: Show ensemble result
    timeline.add({
      duration: 2000,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 0.7);
        drawDecisionBoundary(progress, COLORS.accent, 3);
        
        // Pulse effect on final boundary
        if (progress > 0.9) {
          const pulse = 2 + Math.sin(Date.now() / 300) * 1;
          ctx.strokeStyle = COLORS.highlight;
          ctx.lineWidth = pulse;
          ctx.globalAlpha = 0.5;
          
          // Simplified decision boundary
          ctx.beginPath();
          for (let x = bounds.xMin; x <= bounds.xMax; x += 0.05) {
            const y = Math.sin(x * 2) * 0.5 + 0.2;
            if (x === bounds.xMin) {
              ctx.moveTo(toCanvasX(x), toCanvasY(y));
            } else {
              ctx.lineTo(toCanvasX(x), toCanvasY(y));
            }
          }
          ctx.stroke();
        }
        
        // Draw caption
        ctx.fillStyle = COLORS.text;
        ctx.font = '18px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Ensemble Prediction (Reduced Variance)', width / 2, 30);
      }
    });
    
    // Phase 5: Show variance reduction
    timeline.add({
      duration: 1500,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawGrid();
        drawDataPoints(data, 0.7);
        drawDecisionBoundary(1, COLORS.accent, 3);
        drawVarianceChart(progress);
        
        // Draw final stats
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Number of Estimators: ${params.n_estimators}`, 60, 90);
        ctx.fillText(`Bootstrap Ratio: ${params.bootstrap_ratio}`, 60, 110);
        ctx.fillText(`Base Learner: ${params.base_learner}`, 60, 130);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'bootstrap-process':
      params.show_individuals = false;
      params.show_ensemble = false;
      params.show_variance = false;
      animateBaggingProcess();
      break;
      
    case 'individual-models':
      params.show_bootstraps = false;
      params.show_ensemble = false;
      params.show_variance = false;
      animateBaggingProcess();
      break;
      
    case 'ensemble-result':
      params.show_bootstraps = false;
      params.show_individuals = false;
      params.show_variance = false;
      animateBaggingProcess();
      break;
      
    case 'variance-reduction':
      params.show_bootstraps = false;
      params.show_individuals = false;
      params.show_ensemble = false;
      animateBaggingProcess();
      break;
      
    case 'all':
      animateBaggingProcess();
      break;
      
    default:
      animateBaggingProcess();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Estimators',
        type: 'range',
        min: 1,
        max: 20,
        step: 1,
        value: params.n_estimators,
        onChange: (value) => {
          params.n_estimators = parseInt(value);
          visualizeBagging(containerId, visualizationType, params);
        }
      },
      {
        label: 'Bootstrap Ratio',
        type: 'range',
        min: 0.5,
        max: 1.0,
        step: 0.05,
        value: params.bootstrap_ratio,
        onChange: (value) => {
          params.bootstrap_ratio = parseFloat(value);
          visualizeBagging(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Bootstrap Samples',
        type: 'checkbox',
        checked: params.show_bootstraps,
        onChange: (value) => {
          params.show_bootstraps = value;
          visualizeBagging(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Individual Models',
        type: 'checkbox',
        checked: params.show_individuals,
        onChange: (value) => {
          params.show_individuals = value;
          visualizeBagging(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Variance Chart',
        type: 'checkbox',
        checked: params.show_variance,
        onChange: (value) => {
          params.show_variance = value;
          visualizeBagging(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'bootstrap-process', label: 'Bootstrap Sampling', selected: visualizationType === 'bootstrap-process' },
          { value: 'individual-models', label: 'Individual Models', selected: visualizationType === 'individual-models' },
          { value: 'ensemble-result', label: 'Ensemble Prediction', selected: visualizationType === 'ensemble-result' },
          { value: 'variance-reduction', label: 'Variance Analysis', selected: visualizationType === 'variance-reduction' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizeBagging(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'Bagging Parameters',
      description: 'Adjust parameters to see how they affect the ensemble learning process.'
    });
  }
}

// ==================== ADAVISUALIZATION ====================
function visualizeAdaBoost(containerId, type, params) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Clear previous content
    container.innerHTML = '';
    
    // Create canvas and context
    const canvas = document.createElement('canvas');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    container.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    
    // Get algorithm parameters
    const algoParams = window.algorithmData?.adaboost?.visualization?.parameters || {};
    const finalParams = { ...algoParams, ...params };
    
    const {
        n_samples = 100,
        n_estimators = 10,
        learning_rate = 1.0,
        animation_duration = 2000
    } = finalParams;
    
    // Animation state
    let currentIteration = 0;
    let animationStartTime = null;
    let isAnimating = false;
    
    // Initialize data points
    const points = generateClassificationData(n_samples, 2, 0.3);
    const weights = Array(n_samples).fill(1/n_samples);
    const weakLearners = [];
    const errors = [];
    
    // Main animation function
    function animate(timestamp) {
        if (!animationStartTime) animationStartTime = timestamp;
        const elapsed = timestamp - animationStartTime;
        const progress = Math.min(elapsed / animation_duration, 1);
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw based on current type
        switch(type) {
            case 'weight-evolution':
                drawWeightEvolution(progress);
                break;
            case 'weak-learners':
                drawWeakLearners(progress);
                break;
            case 'ensemble-evolution':
                drawEnsembleEvolution(progress);
                break;
            case 'error-convergence':
                drawErrorConvergence(progress);
                break;
            case 'all':
                drawAllView(progress);
                break;
            default:
                drawWeightEvolution(progress);
        }
        
        // Continue animation if not complete
        if (progress < 1 || currentIteration < n_estimators) {
            requestAnimationFrame(animate);
        } else {
            isAnimating = false;
            // Show final state with subtle pulse
            setTimeout(() => pulseFinalState(), 500);
        }
    }
    
    function drawWeightEvolution(progress) {
        // Draw points with size proportional to weights
        points.forEach((point, i) => {
            const size = 5 + weights[i] * 40; // Scale weight to size
            const alpha = 0.6 + weights[i] * 0.4; // Scale weight to opacity
            
            ctx.globalAlpha = alpha;
            ctx.fillStyle = point.label === 1 ? '#ff6b6b' : '#4ecdc4';
            ctx.beginPath();
            ctx.arc(point.x * canvas.width, point.y * canvas.height, size, 0, Math.PI * 2);
            ctx.fill();
        });
        
        // Show annotation for current iteration
        if (progress > 0.2 && progress < 0.8) {
            ctx.globalAlpha = (Math.sin(progress * Math.PI * 4) + 1) * 0.3;
            ctx.fillStyle = '#2d3436';
            ctx.font = '14px Arial';
            ctx.fillText(`Iteration ${currentIteration + 1}: Weight evolution`, 20, 30);
            ctx.globalAlpha = 1;
        }
    }
    
    function drawWeakLearners(progress) {
        // Draw current weak learner decision boundary
        if (weakLearners[currentIteration]) {
            const learner = weakLearners[currentIteration];
            const alpha = Math.sin(progress * Math.PI);
            
            ctx.globalAlpha = alpha * 0.7;
            ctx.strokeStyle = '#fd79a8';
            ctx.lineWidth = 2;
            
            if (learner.axis === 'x') {
                const x = learner.threshold * canvas.width;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            } else {
                const y = learner.threshold * canvas.height;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
            
            // Label the weak learner
            ctx.globalAlpha = alpha * 0.9;
            ctx.fillStyle = '#fd79a8';
            ctx.font = '12px Arial';
            ctx.fillText(`Weak Learner ${currentIteration + 1}`, 20, 30);
            ctx.globalAlpha = 1;
        }
        
        // Draw points
        drawPointsWithWeights();
    }
    
    function drawEnsembleEvolution(progress) {
        // Draw all weak learners with diminishing opacity
        weakLearners.forEach((learner, idx) => {
            const alpha = 0.2 + (idx / weakLearners.length) * 0.5;
            ctx.globalAlpha = alpha;
            ctx.strokeStyle = '#fd79a8';
            ctx.lineWidth = 1;
            
            if (learner.axis === 'x') {
                const x = learner.threshold * canvas.width;
                ctx.beginPath();
                ctx.moveTo(x, 0);
                ctx.lineTo(x, canvas.height);
                ctx.stroke();
            } else {
                const y = learner.threshold * canvas.height;
                ctx.beginPath();
                ctx.moveTo(0, y);
                ctx.lineTo(canvas.width, y);
                ctx.stroke();
            }
        });
        
        // Draw current ensemble boundary with emphasis
        if (progress > 0.3) {
            const ensembleAlpha = 0.5 + 0.5 * Math.sin(progress * Math.PI * 2);
            ctx.globalAlpha = ensembleAlpha;
            ctx.strokeStyle = '#00b894';
            ctx.lineWidth = 3;
            
            // Simplified ensemble boundary visualization
            ctx.beginPath();
            ctx.moveTo(canvas.width * 0.2, canvas.height * 0.7);
            ctx.bezierCurveTo(
                canvas.width * 0.5, canvas.height * (0.3 + progress * 0.4),
                canvas.width * 0.5, canvas.height * (0.3 + progress * 0.4),
                canvas.width * 0.8, canvas.height * 0.7
            );
            ctx.stroke();
            
            ctx.globalAlpha = 1;
            ctx.fillStyle = '#00b894';
            ctx.font = '14px Arial';
            ctx.fillText('Ensemble Boundary Emerging', 20, 30);
        }
        
        drawPointsWithWeights();
    }
    
    function drawErrorConvergence(progress) {
        // Draw error curve
        if (errors.length > 1) {
            ctx.strokeStyle = '#e17055';
            ctx.lineWidth = 2;
            ctx.beginPath();
            
            errors.forEach((error, idx) => {
                const x = (idx / (n_estimators - 1)) * canvas.width * 0.8 + canvas.width * 0.1;
                const y = canvas.height * 0.8 - error * canvas.height * 0.6;
                
                if (idx === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            // Highlight current error point
            if (currentIteration > 0) {
                const currentX = (currentIteration / (n_estimators - 1)) * canvas.width * 0.8 + canvas.width * 0.1;
                const currentY = canvas.height * 0.8 - errors[currentIteration] * canvas.height * 0.6;
                
                ctx.fillStyle = '#e17055';
                ctx.beginPath();
                ctx.arc(currentX, currentY, 6, 0, Math.PI * 2);
                ctx.fill();
                
                // Pulse animation for current point
                const pulseSize = 6 + 4 * Math.sin(progress * Math.PI * 4);
                ctx.globalAlpha = 0.3;
                ctx.beginPath();
                ctx.arc(currentX, currentY, pulseSize, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1;
                
                // Show error value
                ctx.fillText(`Error: ${errors[currentIteration].toFixed(3)}`, currentX + 10, currentY - 10);
            }
            
            // Labels
            ctx.fillStyle = '#2d3436';
            ctx.fillText('Training Error Convergence', canvas.width * 0.1, canvas.height * 0.1);
            ctx.fillText('Iterations →', canvas.width * 0.4, canvas.height * 0.85);
        }
    }
    
    function drawAllView(progress) {
        // Composite view showing multiple aspects
        const sectionHeight = canvas.height / 3;
        
        // Top section: Weight evolution
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, canvas.width, sectionHeight);
        ctx.clip();
        drawWeightEvolution(progress);
        ctx.restore();
        
        // Middle section: Weak learners
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, sectionHeight, canvas.width, sectionHeight);
        ctx.clip();
        drawWeakLearners(progress);
        ctx.restore();
        
        // Bottom section: Error convergence
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, sectionHeight * 2, canvas.width, sectionHeight);
        ctx.clip();
        drawErrorConvergence(progress);
        ctx.restore();
        
        // Separator lines
        ctx.strokeStyle = '#dfe6e9';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(0, sectionHeight);
        ctx.lineTo(canvas.width, sectionHeight);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, sectionHeight * 2);
        ctx.lineTo(canvas.width, sectionHeight * 2);
        ctx.stroke();
        
        // Title
        ctx.fillStyle = '#2d3436';
        ctx.font = '16px Arial';
        ctx.fillText('AdaBoost Complete View', 20, 25);
    }
    
    function drawPointsWithWeights() {
        points.forEach((point, i) => {
            const size = 4 + weights[i] * 20;
            const alpha = 0.7 + weights[i] * 0.3;
            
            ctx.globalAlpha = alpha;
            ctx.fillStyle = point.label === 1 ? '#ff6b6b' : '#4ecdc4';
            ctx.beginPath();
            ctx.arc(point.x * canvas.width, point.y * canvas.height, size, 0, Math.PI * 2);
            ctx.fill();
        });
        ctx.globalAlpha = 1;
    }
    
    function pulseFinalState() {
        // Create a subtle pulse effect when animation completes
        let pulseProgress = 0;
        const pulseDuration = 1000;
        const startTime = Date.now();
        
        function pulse() {
            const elapsed = Date.now() - startTime;
            pulseProgress = elapsed / pulseDuration;
            
            if (pulseProgress < 1) {
                // Draw with pulse effect
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                // Enhanced final visualization based on type
                switch(type) {
                    case 'weight-evolution':
                        drawWeightEvolution(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'weak-learners':
                        drawWeakLearners(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'ensemble-evolution':
                        drawEnsembleEvolution(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'error-convergence':
                        drawErrorConvergence(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'all':
                        drawAllView(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                }
                
                requestAnimationFrame(pulse);
            }
        }
        
        pulse();
    }
    
    // Initialize and start animation
    isAnimating = true;
    requestAnimationFrame(animate);
}

// ==================== GBM VISUALIZATION ====================
function visualizeGBM(containerId, type, params) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = '';
    const canvas = document.createElement('canvas');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    container.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    
    const algoParams = window.algorithmData?.gbm?.visualization?.parameters || {};
    const finalParams = { ...algoParams, ...params };
    
    const {
        n_samples = 100,
        n_estimators = 10,
        learning_rate = 0.1,
        animation_duration = 2200
    } = finalParams;
    
    // Animation state
    let currentIteration = 0;
    let animationStartTime = null;
    let isAnimating = false;
    
    // Generate regression data
    const {points, trueFunction} = generateRegressionData(n_samples);
    const predictions = Array(n_samples).fill(0);
    const residuals = points.map(point => point.y);
    const models = [];
    const losses = [];
    
    function animate(timestamp) {
        if (!animationStartTime) animationStartTime = timestamp;
        const elapsed = timestamp - animationStartTime;
        const progress = Math.min(elapsed / animation_duration, 1);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        switch(type) {
            case 'residual-fitting':
                drawResidualFitting(progress);
                break;
            case 'prediction-evolution':
                drawPredictionEvolution(progress);
                break;
            case 'loss-reduction':
                drawLossReduction(progress);
                break;
            case 'feature-importance':
                drawFeatureImportance(progress);
                break;
            case 'all':
                drawAllViewGBM(progress);
                break;
            default:
                drawResidualFitting(progress);
        }
        
        if (progress < 1 || currentIteration < n_estimators) {
            requestAnimationFrame(animate);
        } else {
            isAnimating = false;
            setTimeout(() => pulseFinalStateGBM(), 500);
        }
    }
    
    function drawResidualFitting(progress) {
        // Draw true function
        ctx.strokeStyle = '#74b9ff';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        drawFunction(trueFunction, ctx);
        ctx.setLineDash([]);
        
        // Draw current predictions
        ctx.strokeStyle = '#e17055';
        ctx.lineWidth = 3;
        drawFunction(x => predictGBM(x), ctx);
        
        // Draw residuals as vertical lines
        ctx.strokeStyle = '#ff7675';
        ctx.lineWidth = 1;
        points.forEach((point, i) => {
            const predictedY = predictGBM(point.x);
            const screenX = point.x * canvas.width;
            const trueScreenY = (1 - point.y) * canvas.height;
            const predScreenY = (1 - predictedY) * canvas.height;
            
            ctx.beginPath();
            ctx.moveTo(screenX, trueScreenY);
            ctx.lineTo(screenX, predScreenY);
            ctx.stroke();
            
            // Residual value
            ctx.fillStyle = '#ff7675';
            ctx.font = '10px Arial';
            ctx.fillText(residuals[i].toFixed(2), screenX + 5, (trueScreenY + predScreenY) / 2);
        });
        
        // Draw data points
        drawRegressionPoints();
        
        // Annotation
        if (progress > 0.2 && progress < 0.8) {
            ctx.globalAlpha = (Math.sin(progress * Math.PI * 4) + 1) * 0.3;
            ctx.fillStyle = '#2d3436';
            ctx.font = '14px Arial';
            ctx.fillText(`Fitting Residuals - Iteration ${currentIteration + 1}`, 20, 30);
            ctx.globalAlpha = 1;
        }
    }
    
    function drawPredictionEvolution(progress) {
        // Draw all previous predictions with diminishing opacity
        for (let i = 0; i < models.length; i++) {
            const alpha = 0.1 + (i / models.length) * 0.3;
            ctx.globalAlpha = alpha;
            ctx.strokeStyle = '#e17055';
            ctx.lineWidth = 1;
            drawFunction(x => predictGBMUpTo(x, i), ctx);
        }
        
        // Draw current prediction with emphasis
        ctx.globalAlpha = 0.5 + 0.5 * Math.sin(progress * Math.PI);
        ctx.strokeStyle = '#00b894';
        ctx.lineWidth = 3;
        drawFunction(x => predictGBM(x), ctx);
        ctx.globalAlpha = 1;
        
        // Draw true function
        ctx.strokeStyle = '#74b9ff';
        ctx.lineWidth = 2;
        ctx.setLineDash([5, 5]);
        drawFunction(trueFunction, ctx);
        ctx.setLineDash([]);
        
        drawRegressionPoints();
        
        // Label
        ctx.fillStyle = '#00b894';
        ctx.font = '14px Arial';
        ctx.fillText('Prediction Evolution', 20, 30);
    }
    
    function drawLossReduction(progress) {
        // Draw loss curve
        if (losses.length > 1) {
            ctx.strokeStyle = '#6c5ce7';
            ctx.lineWidth = 3;
            ctx.beginPath();
            
            losses.forEach((loss, idx) => {
                const x = (idx / (n_estimators - 1)) * canvas.width * 0.8 + canvas.width * 0.1;
                const y = canvas.height * 0.8 - Math.log(loss + 1) * canvas.height * 0.6;
                
                if (idx === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            // Highlight current loss
            if (currentIteration > 0) {
                const currentX = (currentIteration / (n_estimators - 1)) * canvas.width * 0.8 + canvas.width * 0.1;
                const currentY = canvas.height * 0.8 - Math.log(losses[currentIteration] + 1) * canvas.height * 0.6;
                
                ctx.fillStyle = '#6c5ce7';
                ctx.beginPath();
                ctx.arc(currentX, currentY, 6, 0, Math.PI * 2);
                ctx.fill();
                
                // Pulse animation
                const pulseSize = 6 + 4 * Math.sin(progress * Math.PI * 4);
                ctx.globalAlpha = 0.3;
                ctx.beginPath();
                ctx.arc(currentX, currentY, pulseSize, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1;
                
                // Show loss value
                ctx.fillText(`Loss: ${losses[currentIteration].toFixed(4)}`, currentX + 10, currentY - 10);
            }
            
            // Labels
            ctx.fillStyle = '#2d3436';
            ctx.fillText('Loss Reduction', canvas.width * 0.1, canvas.height * 0.1);
            ctx.fillText('Iterations →', canvas.width * 0.4, canvas.height * 0.85);
            ctx.fillText('Log(Loss)', canvas.width * 0.02, canvas.height * 0.4);
        }
    }
    
    function drawFeatureImportance(progress) {
        // Simplified feature importance visualization
        const features = ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'];
        const importance = features.map((_, i) => 
            Math.sin((i + 1) * 0.5 + currentIteration * 0.2) * 0.5 + 0.5
        );
        
        const barWidth = canvas.width * 0.8 / features.length;
        const maxBarHeight = canvas.height * 0.6;
        
        features.forEach((feature, i) => {
            const barHeight = importance[i] * maxBarHeight * progress;
            const x = canvas.width * 0.1 + i * barWidth;
            const y = canvas.height * 0.8 - barHeight;
            
            // Bar with gradient
            const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
            gradient.addColorStop(0, '#00cec9');
            gradient.addColorStop(1, '#0984e3');
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, barWidth * 0.8, barHeight);
            
            // Feature label
            ctx.fillStyle = '#2d3436';
            ctx.font = '12px Arial';
            ctx.fillText(feature, x, canvas.height * 0.85);
            
            // Importance value
            ctx.fillText(importance[i].toFixed(2), x, y - 5);
        });
        
        // Title
        ctx.fillStyle = '#2d3436';
        ctx.font = '16px Arial';
        ctx.fillText('Feature Importance', canvas.width * 0.1, canvas.height * 0.1);
    }
    
    function drawAllViewGBM(progress) {
        const sectionHeight = canvas.height / 2;
        const sectionWidth = canvas.width / 2;
        
        // Top-left: Residual fitting
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, sectionWidth, sectionHeight);
        ctx.clip();
        drawResidualFitting(progress);
        ctx.restore();
        
        // Top-right: Prediction evolution
        ctx.save();
        ctx.beginPath();
        ctx.rect(sectionWidth, 0, sectionWidth, sectionHeight);
        ctx.clip();
        drawPredictionEvolution(progress);
        ctx.restore();
        
        // Bottom-left: Loss reduction
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, sectionHeight, sectionWidth, sectionHeight);
        ctx.clip();
        drawLossReduction(progress);
        ctx.restore();
        
        // Bottom-right: Feature importance
        ctx.save();
        ctx.beginPath();
        ctx.rect(sectionWidth, sectionHeight, sectionWidth, sectionHeight);
        ctx.clip();
        drawFeatureImportance(progress);
        ctx.restore();
        
        // Separator lines
        ctx.strokeStyle = '#dfe6e9';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sectionWidth, 0);
        ctx.lineTo(sectionWidth, canvas.height);
        ctx.moveTo(0, sectionHeight);
        ctx.lineTo(canvas.width, sectionHeight);
        ctx.stroke();
        
        // Title
        ctx.fillStyle = '#2d3436';
        ctx.font = '16px Arial';
        ctx.fillText('GBM Complete View', 20, 25);
    }
    
    function drawRegressionPoints() {
        points.forEach(point => {
            ctx.fillStyle = '#fd79a8';
            ctx.beginPath();
            ctx.arc(point.x * canvas.width, (1 - point.y) * canvas.height, 4, 0, Math.PI * 2);
            ctx.fill();
        });
    }
    
    function drawFunction(func, context) {
        context.beginPath();
        for (let x = 0; x <= canvas.width; x += 2) {
            const normalizedX = x / canvas.width;
            const normalizedY = func(normalizedX);
            const screenY = (1 - normalizedY) * canvas.height;
            
            if (x === 0) {
                context.moveTo(x, screenY);
            } else {
                context.lineTo(x, screenY);
            }
        }
        context.stroke();
    }
    
    function predictGBM(x) {
        return models.reduce((sum, model) => sum + model.predict(x) * learning_rate, 0);
    }
    
    function predictGBMUpTo(x, iteration) {
        return models.slice(0, iteration + 1).reduce((sum, model) => sum + model.predict(x) * learning_rate, 0);
    }
    
    function pulseFinalStateGBM() {
        // Pulse effect for GBM completion
        let pulseProgress = 0;
        const pulseDuration = 1000;
        const startTime = Date.now();
        
        function pulse() {
            const elapsed = Date.now() - startTime;
            pulseProgress = elapsed / pulseDuration;
            
            if (pulseProgress < 1) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                switch(type) {
                    case 'residual-fitting':
                        drawResidualFitting(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'prediction-evolution':
                        drawPredictionEvolution(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'loss-reduction':
                        drawLossReduction(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'feature-importance':
                        drawFeatureImportance(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'all':
                        drawAllViewGBM(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                }
                
                requestAnimationFrame(pulse);
            }
        }
        
        pulse();
    }
    
    isAnimating = true;
    requestAnimationFrame(animate);
}

// ==================== XGBOOST VISUALIZATION ====================
function visualizeXGBoost(containerId, type, params) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    container.innerHTML = '';
    const canvas = document.createElement('canvas');
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    container.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    
    const algoParams = window.algorithmData?.xgboost?.visualization?.parameters || {};
    const finalParams = { ...algoParams, ...params };
    
    const {
        n_estimators = 50,
        learning_rate = 0.1,
        animation_duration = 3000
    } = finalParams;
    
    // Animation state
    let currentIteration = 0;
    let animationStartTime = null;
    let isAnimating = false;
    
    // Generate classification data
    const points = generateClassificationData(100, 2, 0.2);
    const trees = [];
    const errors = [];
    const featureImportance = {f0: 0, f1: 0, f2: 0, f3: 0};
    
    function animate(timestamp) {
        if (!animationStartTime) animationStartTime = timestamp;
        const elapsed = timestamp - animationStartTime;
        const progress = Math.min(elapsed / animation_duration, 1);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        switch(type) {
            case 'boosting-process':
                drawBoostingProcess(progress);
                break;
            case 'loss-reduction':
                drawLossReductionXGB(progress);
                break;
            case 'feature-importance':
                drawFeatureImportanceXGB(progress);
                break;
            case 'tree-structure':
                drawTreeStructure(progress);
                break;
            case 'all':
                drawAllViewXGB(progress);
                break;
            default:
                drawBoostingProcess(progress);
        }
        
        if (progress < 1 || currentIteration < n_estimators) {
            requestAnimationFrame(animate);
        } else {
            isAnimating = false;
            setTimeout(() => pulseFinalStateXGB(), 500);
        }
    }
    
    function drawBoostingProcess(progress) {
        // Draw decision boundary evolution
        if (trees.length > 0) {
            // Draw all previous boundaries with diminishing opacity
            for (let i = 0; i < trees.length; i++) {
                const alpha = 0.1 + (i / trees.length) * 0.2;
                ctx.globalAlpha = alpha;
                ctx.strokeStyle = '#74b9ff';
                ctx.lineWidth = 1;
                drawDecisionBoundary(i);
            }
            
            // Draw current boundary with emphasis
            const alpha = 0.5 + 0.5 * Math.sin(progress * Math.PI);
            ctx.globalAlpha = alpha;
            ctx.strokeStyle = '#00b894';
            ctx.lineWidth = 3;
            drawDecisionBoundary(currentIteration);
            ctx.globalAlpha = 1;
        }
        
        // Draw data points
        drawClassificationPoints();
        
        // Annotation
        if (progress > 0.2 && progress < 0.8) {
            ctx.globalAlpha = (Math.sin(progress * Math.PI * 4) + 1) * 0.3;
            ctx.fillStyle = '#2d3436';
            ctx.font = '14px Arial';
            ctx.fillText(`Boosting Round ${currentIteration + 1}/${n_estimators}`, 20, 30);
            ctx.globalAlpha = 1;
        }
    }
    
    function drawLossReductionXGB(progress) {
        // Draw error curve with early stopping highlight
        if (errors.length > 1) {
            ctx.strokeStyle = '#6c5ce7';
            ctx.lineWidth = 3;
            ctx.beginPath();
            
            errors.forEach((error, idx) => {
                const x = (idx / (n_estimators - 1)) * canvas.width * 0.8 + canvas.width * 0.1;
                const y = canvas.height * 0.8 - error * canvas.height * 0.6;
                
                if (idx === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            });
            
            ctx.stroke();
            
            // Highlight best iteration (early stopping point)
            const bestIteration = errors.indexOf(Math.min(...errors));
            if (bestIteration > 0) {
                const bestX = (bestIteration / (n_estimators - 1)) * canvas.width * 0.8 + canvas.width * 0.1;
                const bestY = canvas.height * 0.8 - errors[bestIteration] * canvas.height * 0.6;
                
                ctx.fillStyle = '#00b894';
                ctx.beginPath();
                ctx.arc(bestX, bestY, 8, 0, Math.PI * 2);
                ctx.fill();
                
                // Early stopping annotation
                ctx.fillStyle = '#00b894';
                ctx.font = '12px Arial';
                ctx.fillText('Early Stopping Point', bestX + 10, bestY - 10);
            }
            
            // Highlight current error
            if (currentIteration > 0) {
                const currentX = (currentIteration / (n_estimators - 1)) * canvas.width * 0.8 + canvas.width * 0.1;
                const currentY = canvas.height * 0.8 - errors[currentIteration] * canvas.height * 0.6;
                
                ctx.fillStyle = '#6c5ce7';
                ctx.beginPath();
                ctx.arc(currentX, currentY, 6, 0, Math.PI * 2);
                ctx.fill();
                
                // Pulse animation
                const pulseSize = 6 + 4 * Math.sin(progress * Math.PI * 4);
                ctx.globalAlpha = 0.3;
                ctx.beginPath();
                ctx.arc(currentX, currentY, pulseSize, 0, Math.PI * 2);
                ctx.fill();
                ctx.globalAlpha = 1;
                
                // Show error value
                ctx.fillText(`Error: ${errors[currentIteration].toFixed(4)}`, currentX + 10, currentY - 10);
            }
            
            // Labels
            ctx.fillStyle = '#2d3436';
            ctx.fillText('Error Reduction with Early Stopping', canvas.width * 0.1, canvas.height * 0.1);
            ctx.fillText('Iterations →', canvas.width * 0.4, canvas.height * 0.85);
        }
    }
    
    function drawFeatureImportanceXGB(progress) {
        // Calculate feature importance based on usage in trees
        const features = Object.keys(featureImportance);
        const importanceValues = features.map(f => featureImportance[f]);
        const maxImportance = Math.max(...importanceValues);
        const normalizedImportance = importanceValues.map(v => v / (maxImportance || 1));
        
        const barWidth = canvas.width * 0.8 / features.length;
        const maxBarHeight = canvas.height * 0.6;
        
        features.forEach((feature, i) => {
            const barHeight = normalizedImportance[i] * maxBarHeight * progress;
            const x = canvas.width * 0.1 + i * barWidth;
            const y = canvas.height * 0.8 - barHeight;
            
            // Bar with gradient
            const gradient = ctx.createLinearGradient(x, y, x, y + barHeight);
            gradient.addColorStop(0, '#fd79a8');
            gradient.addColorStop(1, '#e84393');
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, barWidth * 0.8, barHeight);
            
            // Feature label
            ctx.fillStyle = '#2d3436';
            ctx.font = '12px Arial';
            ctx.fillText(feature, x, canvas.height * 0.85);
            
            // Importance value
            ctx.fillText(importanceValues[i].toFixed(1), x, y - 5);
        });
        
        // Title
        ctx.fillStyle = '#2d3436';
        ctx.font = '16px Arial';
        ctx.fillText('XGBoost Feature Importance', canvas.width * 0.1, canvas.height * 0.1);
    }
    
    function drawTreeStructure(progress) {
        // Draw a simplified tree structure
        const centerX = canvas.width / 2;
        const startY = canvas.height * 0.1;
        const levelHeight = canvas.height * 0.15;
        
        // Draw root node
        drawTreeNode(centerX, startY, 'Root', progress);
        
        // Draw child nodes with animation
        if (progress > 0.3) {
            const childProgress = (progress - 0.3) / 0.7;
            drawTreeNode(centerX - 100, startY + levelHeight, 'Feature 1 < 0.5', childProgress);
            drawTreeNode(centerX + 100, startY + levelHeight, 'Feature 2 ≥ 0.3', childProgress);
            
            // Connect nodes
            ctx.strokeStyle = '#636e72';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(centerX, startY + 15);
            ctx.lineTo(centerX - 100, startY + levelHeight - 15);
            ctx.moveTo(centerX, startY + 15);
            ctx.lineTo(centerX + 100, startY + levelHeight - 15);
            ctx.stroke();
            
            // Draw leaves with further animation
            if (progress > 0.6) {
                const leafProgress = (progress - 0.6) / 0.4;
                drawTreeNode(centerX - 150, startY + levelHeight * 2, 'Leaf: Class 0', leafProgress);
                drawTreeNode(centerX - 50, startY + levelHeight * 2, 'Leaf: Class 1', leafProgress);
                drawTreeNode(centerX + 50, startY + levelHeight * 2, 'Leaf: Class 0', leafProgress);
                drawTreeNode(centerX + 150, startY + levelHeight * 2, 'Leaf: Class 1', leafProgress);
                
                // Connect to parent nodes
                ctx.beginPath();
                ctx.moveTo(centerX - 100, startY + levelHeight + 15);
                ctx.lineTo(centerX - 150, startY + levelHeight * 2 - 15);
                ctx.moveTo(centerX - 100, startY + levelHeight + 15);
                ctx.lineTo(centerX - 50, startY + levelHeight * 2 - 15);
                ctx.moveTo(centerX + 100, startY + levelHeight + 15);
                ctx.lineTo(centerX + 50, startY + levelHeight * 2 - 15);
                ctx.moveTo(centerX + 100, startY + levelHeight + 15);
                ctx.lineTo(centerX + 150, startY + levelHeight * 2 - 15);
                ctx.stroke();
            }
        }
        
        // Title
        ctx.fillStyle = '#2d3436';
        ctx.font = '16px Arial';
        ctx.fillText('XGBoost Tree Structure', canvas.width * 0.1, 25);
    }
    
    function drawTreeNode(x, y, label, progress) {
        const nodeSize = 30 * progress;
        const alpha = 0.5 + 0.5 * progress;
        
        ctx.globalAlpha = alpha;
        ctx.fillStyle = '#74b9ff';
        ctx.beginPath();
        ctx.arc(x, y, nodeSize, 0, Math.PI * 2);
        ctx.fill();
        
        ctx.strokeStyle = '#2d3436';
        ctx.lineWidth = 2;
        ctx.stroke();
        
        ctx.fillStyle = '#2d3436';
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(label, x, y + 4);
        ctx.textAlign = 'left';
        ctx.globalAlpha = 1;
    }
    
    function drawAllViewXGB(progress) {
        const sectionHeight = canvas.height / 2;
        const sectionWidth = canvas.width / 2;
        
        // Top-left: Boosting process
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, sectionWidth, sectionHeight);
        ctx.clip();
        drawBoostingProcess(progress);
        ctx.restore();
        
        // Top-right: Loss reduction
        ctx.save();
        ctx.beginPath();
        ctx.rect(sectionWidth, 0, sectionWidth, sectionHeight);
        ctx.clip();
        drawLossReductionXGB(progress);
        ctx.restore();
        
        // Bottom-left: Feature importance
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, sectionHeight, sectionWidth, sectionHeight);
        ctx.clip();
        drawFeatureImportanceXGB(progress);
        ctx.restore();
        
        // Bottom-right: Tree structure
        ctx.save();
        ctx.beginPath();
        ctx.rect(sectionWidth, sectionHeight, sectionWidth, sectionHeight);
        ctx.clip();
        drawTreeStructure(progress);
        ctx.restore();
        
        // Separator lines
        ctx.strokeStyle = '#dfe6e9';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(sectionWidth, 0);
        ctx.lineTo(sectionWidth, canvas.height);
        ctx.moveTo(0, sectionHeight);
        ctx.lineTo(canvas.width, sectionHeight);
        ctx.stroke();
        
        // Title
        ctx.fillStyle = '#2d3436';
        ctx.font = '16px Arial';
        ctx.fillText('XGBoost Complete View', 20, 25);
    }
    
    function drawDecisionBoundary(upToIteration) {
        // Simplified decision boundary visualization
        ctx.beginPath();
        for (let x = 0; x <= canvas.width; x += 5) {
            for (let y = 0; y <= canvas.height; y += 5) {
                const normX = x / canvas.width;
                const normY = y / canvas.height;
                
                // Simplified prediction based on position
                const prediction = (normX - 0.5) * (normY - 0.5) > 0 ? 1 : 0;
                
                if (prediction === 1) {
                    if (y === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                }
            }
        }
        ctx.stroke();
    }
    
    function drawClassificationPoints() {
        points.forEach(point => {
            ctx.fillStyle = point.label === 1 ? '#ff6b6b' : '#4ecdc4';
            ctx.beginPath();
            ctx.arc(point.x * canvas.width, point.y * canvas.height, 5, 0, Math.PI * 2);
            ctx.fill();
        });
    }
    
    function pulseFinalStateXGB() {
        // Pulse effect for XGBoost completion
        let pulseProgress = 0;
        const pulseDuration = 1000;
        const startTime = Date.now();
        
        function pulse() {
            const elapsed = Date.now() - startTime;
            pulseProgress = elapsed / pulseDuration;
            
            if (pulseProgress < 1) {
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                
                switch(type) {
                    case 'boosting-process':
                        drawBoostingProcess(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'loss-reduction':
                        drawLossReductionXGB(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'feature-importance':
                        drawFeatureImportanceXGB(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'tree-structure':
                        drawTreeStructure(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                    case 'all':
                        drawAllViewXGB(0.5 + 0.5 * Math.sin(pulseProgress * Math.PI));
                        break;
                }
                
                requestAnimationFrame(pulse);
            }
        }
        
        pulse();
    }
    
    isAnimating = true;
    requestAnimationFrame(animate);
}

// =============================================
// Enhanced Neural Network Visualizations
// =============================================
function visualizeNeuralNetwork(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    n_layers: 3,
    layer_sizes: [4, 8, 2],
    activation: 'relu',
    learning_rate: 0.01,
    epochs: 100,
    batch_size: 32,
    show_weights: true,
    show_activations: false,
    show_gradients: false,
    animation_duration: 3000,
    interactive: true,
    controlsContainer: null,
    show_grid: true,
    show_axes: true,
    data_distribution: 'spiral'
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 1000;
  const height = 600;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // Generate data for visualization
  const data = DataSimulator.generateNeuralNetworkData({
    n_samples: 200,
    n_classes: params.layer_sizes[params.layer_sizes.length - 1],
    complexity: 2,
    distribution: params.data_distribution
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
  const toCanvasX = (x) => MathUtils.map(x, bounds.xMin, bounds.xMax, 50, width - 350);
  const toCanvasY = (y) => MathUtils.map(y, bounds.yMin, bounds.yMax, height - 50, 50);
  
  // Neural Network Model
  class NeuralNetwork {
    constructor(layerSizes, activation = 'relu', learningRate = 0.01) {
      this.layerSizes = layerSizes;
      this.activation = activation;
      this.learningRate = learningRate;
      this.weights = [];
      this.biases = [];
      this.activations = [];
      this.gradients = [];
      this.lossHistory = [];
      
      // Initialize weights and biases
      for (let i = 1; i < layerSizes.length; i++) {
        this.weights.push(this.initializeWeights(layerSizes[i-1], layerSizes[i]));
        this.biases.push(new Array(layerSizes[i]).fill(0.1));
      }
    }
    
    initializeWeights(rows, cols) {
      const weights = [];
      for (let i = 0; i < rows; i++) {
        weights.push(new Array(cols).fill(0).map(() => MathUtils.random(-1, 1) * Math.sqrt(2 / rows)));
      }
      return weights;
    }
    
    activate(x) {
      switch (this.activation) {
        case 'sigmoid':
          return MathUtils.sigmoid(x);
        case 'tanh':
          return Math.tanh(x);
        case 'relu':
          return Math.max(0, x);
        default:
          return Math.max(0, x); // Default to ReLU
      }
    }
    
    activateDerivative(x) {
      switch (this.activation) {
        case 'sigmoid':
          const sig = MathUtils.sigmoid(x);
          return sig * (1 - sig);
        case 'tanh':
          return 1 - Math.pow(Math.tanh(x), 2);
        case 'relu':
          return x > 0 ? 1 : 0;
        default:
          return x > 0 ? 1 : 0; // Default to ReLU derivative
      }
    }
    
    forwardPass(inputs) {
      this.activations = [inputs];
      
      for (let i = 0; i < this.weights.length; i++) {
        const layerOutputs = [];
        for (let j = 0; j < this.weights[i][0].length; j++) {
          let sum = this.biases[i][j];
          for (let k = 0; k < inputs.length; k++) {
            sum += inputs[k] * this.weights[i][k][j];
          }
          layerOutputs.push(this.activate(sum));
        }
        this.activations.push(layerOutputs);
        inputs = layerOutputs;
      }
      
      return this.activations[this.activations.length - 1];
    }
    
    backwardPass(inputs, targets) {
      // Calculate output layer error
      const output = this.activations[this.activations.length - 1];
      const outputError = output.map((o, i) => o - targets[i]);
      
      // Calculate gradients
      this.gradients = [];
      let errors = outputError;
      
      for (let i = this.weights.length - 1; i >= 0; i--) {
        const layerGradients = [];
        const prevActivations = this.activations[i];
        const currentActivations = this.activations[i + 1];
        
        for (let j = 0; j < this.weights[i].length; j++) {
          const neuronGradients = [];
          for (let k = 0; k < this.weights[i][j].length; k++) {
            const gradient = errors[k] * this.activateDerivative(currentActivations[k]) * prevActivations[j];
            neuronGradients.push(gradient);
          }
          layerGradients.push(neuronGradients);
        }
        
        this.gradients.unshift(layerGradients);
        
        // Calculate error for next layer
        if (i > 0) {
          const newErrors = new Array(this.weights[i-1][0].length).fill(0);
          for (let j = 0; j < this.weights[i].length; j++) {
            for (let k = 0; k < this.weights[i][j].length; k++) {
              newErrors[j] += errors[k] * this.activateDerivative(currentActivations[k]) * this.weights[i][j][k];
            }
          }
          errors = newErrors;
        }
      }
      
      return outputError;
    }
    
    updateWeights() {
      for (let i = 0; i < this.weights.length; i++) {
        for (let j = 0; j < this.weights[i].length; j++) {
          for (let k = 0; k < this.weights[i][j].length; k++) {
            this.weights[i][j][k] -= this.learningRate * this.gradients[i][j][k];
          }
        }
        
        // Update biases
        for (let k = 0; k < this.biases[i].length; k++) {
          let biasGradient = 0;
          for (let j = 0; j < this.weights[i].length; j++) {
            biasGradient += this.gradients[i][j][k];
          }
          this.biases[i][k] -= this.learningRate * biasGradient / this.weights[i].length;
        }
      }
    }
    
    calculateLoss(output, target) {
      return output.reduce((sum, o, i) => sum + Math.pow(o - target[i], 2), 0) / output.length;
    }
    
    train(data, epochs) {
      for (let epoch = 0; epoch < epochs; epoch++) {
        let totalLoss = 0;
        
        for (const point of data) {
          const inputs = [point.x, point.y];
          const target = new Array(this.layerSizes[this.layerSizes.length - 1]).fill(0);
          target[point.label] = 1;
          
          this.forwardPass(inputs);
          const error = this.backwardPass(inputs, target);
          this.updateWeights();
          
          totalLoss += this.calculateLoss(this.activations[this.activations.length - 1], target);
        }
        
        this.lossHistory.push(totalLoss / data.length);
      }
    }
    
    predict(point) {
      const outputs = this.forwardPass([point.x, point.y]);
      return outputs.indexOf(Math.max(...outputs));
    }
  }
  
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
      ctx.fillText('Feature 1 (X)', width - 350 - 20, toCanvasY(0) - 10);
      ctx.textAlign = 'right';
      ctx.fillText('Feature 2 (Y)', toCanvasX(0) - 10, 30);
    }
    
    ctx.restore();
  };
  
  // Enhanced data points drawing
  const drawDataPoints = (points, progress = 1, highlight = null) => {
    ctx.save();
    
    // Group points by class for sequential animation
    const pointsByClass = {};
    points.forEach(point => {
      if (!pointsByClass[point.label]) pointsByClass[point.label] = [];
      pointsByClass[point.label].push(point);
    });
    
    // Draw each class with different timing
    Object.keys(pointsByClass).forEach((classLabel, classIndex) => {
      const classPoints = pointsByClass[classLabel];
      const classProgress = MathUtils.clamp((progress - classIndex * 0.2) * 1.25, 0, 1);
      
      classPoints.forEach(point => {
        // Dim points that are not in the current highlight
        let alpha = classProgress;
        if (highlight && !highlight.includes(point)) {
          alpha *= 0.3;
        }
        
        ctx.globalAlpha = alpha;
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
    });
    
    ctx.restore();
  };
  
  // Draw neural network architecture
  const drawNetworkArchitecture = (network, progress = 1, currentLayer = 0) => {
    ctx.save();
    
    // Network area
    const networkX = width - 320;
    const networkY = 50;
    const networkWidth = 300;
    const networkHeight = height - 100;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(networkX, networkY, networkWidth, networkHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(networkX, networkY, networkWidth, networkHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '16px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Neural Network Architecture', networkX + networkWidth / 2, networkY - 10);
    
    // Calculate node positions
    const layerSpacing = networkWidth / (network.layerSizes.length + 1);
    const maxNodes = Math.max(...network.layerSizes);
    const nodeRadius = 15;
    
    // Draw layers
    for (let layerIdx = 0; layerIdx < network.layerSizes.length; layerIdx++) {
      const layerProgress = MathUtils.clamp((progress - layerIdx * 0.2) * 1.25, 0, 1);
      const x = networkX + (layerIdx + 1) * layerSpacing;
      const numNodes = network.layerSizes[layerIdx];
      const nodeSpacing = networkHeight / (numNodes + 1);
      
      // Draw layer label
      ctx.fillStyle = COLORS.text;
      ctx.font = '12px Arial';
      ctx.textAlign = 'center';
      ctx.fillText(
        layerIdx === 0 ? 'Input' : 
        layerIdx === network.layerSizes.length - 1 ? 'Output' : `Hidden ${layerIdx}`,
        x, networkY + networkHeight + 20
      );
      
      // Draw nodes
      for (let nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
        const y = networkY + (nodeIdx + 1) * nodeSpacing;
        const isCurrent = layerIdx === currentLayer;
        
        // Draw node
        ctx.beginPath();
        ctx.arc(x, y, nodeRadius, 0, Math.PI * 2);
        
        if (isCurrent) {
          // Highlight current layer nodes
          ctx.fillStyle = COLORS.highlight;
          ctx.fill();
          ctx.strokeStyle = COLORS.text;
          ctx.lineWidth = 2;
          ctx.stroke();
        } else {
          // Regular node
          ctx.fillStyle = '#ffffff';
          ctx.fill();
          ctx.strokeStyle = COLORS.text;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
        
        // Draw node label
        ctx.fillStyle = COLORS.text;
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`${layerIdx}-${nodeIdx}`, x, y);
        
        // Draw activation value if available
        if (network.activations[layerIdx] && nodeIdx < network.activations[layerIdx].length) {
          ctx.font = '8px Arial';
          ctx.fillText(network.activations[layerIdx][nodeIdx].toFixed(2), x, y + 15);
        }
      }
      
      // Draw connections between layers
      if (layerIdx > 0 && layerProgress > 0.5) {
        const prevNumNodes = network.layerSizes[layerIdx - 1];
        const prevX = networkX + layerIdx * layerSpacing;
        const prevNodeSpacing = networkHeight / (prevNumNodes + 1);
        
        for (let prevNodeIdx = 0; prevNodeIdx < prevNumNodes; prevNodeIdx++) {
          const prevY = networkY + (prevNodeIdx + 1) * prevNodeSpacing;
          
          for (let nodeIdx = 0; nodeIdx < numNodes; nodeIdx++) {
            const y = networkY + (nodeIdx + 1) * nodeSpacing;
            
            // Draw connection with weight-based thickness
            ctx.beginPath();
            ctx.moveTo(prevX + nodeRadius, prevY);
            ctx.lineTo(x - nodeRadius, y);
            
            if (network.weights[layerIdx - 1] && network.weights[layerIdx - 1][prevNodeIdx]) {
              const weight = network.weights[layerIdx - 1][prevNodeIdx][nodeIdx];
              const thickness = MathUtils.map(Math.abs(weight), 0, 2, 1, 5);
              const alpha = MathUtils.map(Math.abs(weight), 0, 2, 0.2, 0.8);
              
              ctx.strokeStyle = weight > 0 ? 
                `rgba(76, 175, 80, ${alpha})` : // Green for positive
                `rgba(244, 67, 54, ${alpha})`;  // Red for negative
              
              ctx.lineWidth = thickness;
              ctx.stroke();
              
              // Draw weight value
              if (progress > 0.8) {
                const midX = (prevX + x) / 2;
                const midY = (prevY + y) / 2;
                
                ctx.fillStyle = COLORS.text;
                ctx.font = '8px Arial';
                ctx.textAlign = 'center';
                ctx.fillText(weight.toFixed(2), midX, midY);
              }
            }
          }
        }
      }
    }
    
    ctx.restore();
  };
  
  // Draw decision boundary
  const drawDecisionBoundary = (network, progress = 1) => {
    ctx.save();
    
    // Draw decision regions
    const resolution = 40;
    const cellWidth = (width - 350 - 100) / resolution;
    const cellHeight = (height - 100) / resolution;
    
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x = bounds.xMin + (bounds.xMax - bounds.xMin) * (i / resolution);
        const y = bounds.yMin + (bounds.yMax - bounds.yMin) * (j / resolution);
        
        const predictedClass = network.predict({ x, y });
        const color = COLORS.spectrum[predictedClass % COLORS.spectrum.length];
        
        // Wave effect based on distance from center
        const centerX = resolution / 2;
        const centerY = resolution / 2;
        const distance = Math.sqrt((i - centerX) ** 2 + (j - centerY) ** 2);
        const waveProgress = MathUtils.clamp(progress * 2 - distance / resolution, 0, 1);
        
        ctx.fillStyle = color + Math.floor(50 * waveProgress).toString(16).padStart(2, '0');
        ctx.fillRect(
          toCanvasX(x) - cellWidth / 2,
          toCanvasY(y) - cellHeight / 2,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Draw loss curve
  const drawLossCurve = (lossHistory, progress = 1) => {
    if (lossHistory.length === 0) return;
    
    ctx.save();
    
    // Create a separate area for the loss curve
    const lossWidth = 300;
    const lossHeight = 150;
    const lossX = width - lossWidth - 20;
    const lossY = 20;
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(lossX, lossY, lossWidth, lossHeight);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(lossX, lossY, lossWidth, lossHeight);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Loss Curve', lossX + lossWidth / 2, lossY - 5);
    
    // Find min and max loss for scaling
    const maxLoss = Math.max(...lossHistory);
    const minLoss = Math.min(...lossHistory);
    const lossRange = maxLoss - minLoss || 1; // Avoid division by zero
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(lossX + 30, lossY + 20);
    ctx.lineTo(lossX + 30, lossY + lossHeight - 20);
    ctx.lineTo(lossX + lossWidth - 20, lossY + lossHeight - 20);
    ctx.stroke();
    
    // Draw labels
    ctx.textAlign = 'right';
    ctx.fillText(minLoss.toFixed(2), lossX + 25, lossY + lossHeight - 20);
    ctx.fillText(maxLoss.toFixed(2), lossX + 25, lossY + 20);
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', lossX + lossWidth / 2, lossY + lossHeight);
    
    // Draw loss curve
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const pointsToShow = Math.floor(lossHistory.length * progress);
    const visibleLosses = lossHistory.slice(0, pointsToShow);
    
    visibleLosses.forEach((loss, i) => {
      const x = lossX + 30 + (lossWidth - 50) * (i / (lossHistory.length - 1));
      const y = lossY + lossHeight - 20 - (lossHeight - 40) * ((loss - minLoss) / lossRange);
      
      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    
    ctx.stroke();
    
    // Draw current point
    if (pointsToShow > 0) {
      const currentLoss = lossHistory[pointsToShow - 1];
      const x = lossX + 30 + (lossWidth - 50) * ((pointsToShow - 1) / (lossHistory.length - 1));
      const y = lossY + lossHeight - 20 - (lossHeight - 40) * ((currentLoss - minLoss) / lossRange);
      
      ctx.fillStyle = COLORS.accent;
      ctx.beginPath();
      ctx.arc(x, y, 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw current loss value
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(currentLoss.toFixed(4), x + 5, y);
    }
    
    ctx.restore();
  };
  
  // Animate the neural network training with enhanced cinematic effects
  const animateNeuralNetwork = () => {
    const network = new NeuralNetwork(params.layer_sizes, params.activation, params.learning_rate);
    const epochs = params.epochs;
    const batchSize = 5; // Update in batches for smoother animation
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Animate points appearing class by class
    const classes = [...new Set(data.map(p => p.label))];
    
    classes.forEach((classLabel, classIndex) => {
      timeline.add({
        duration: 800,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawGrid();
          
          // Draw points from previous classes
          for (let i = 0; i < classIndex; i++) {
            const classPoints = data.filter(p => p.label === classes[i]);
            drawDataPoints(classPoints, 1);
          }
          
          // Draw points from current class with animation
          const classPoints = data.filter(p => p.label === classLabel);
          drawDataPoints(classPoints, progress);
          
          // Add class label
          if (progress > 0.8) {
            const centerX = classPoints.reduce((sum, p) => sum + p.x, 0) / classPoints.length;
            const centerY = classPoints.reduce((sum, p) => sum + p.y, 0) / classPoints.length;
            
            ctx.save();
            ctx.fillStyle = COLORS.text;
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText(`Class ${classLabel}`, toCanvasX(centerX), toCanvasY(centerY) - 15);
            ctx.restore();
          }
        }
      }, { delay: classIndex * 300 });
    });
    
    // Phase 2: Animate network architecture building
    for (let layerIdx = 0; layerIdx < params.layer_sizes.length; layerIdx++) {
      timeline.add({
        duration: 1000,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          
          // Draw grid and points
          drawGrid();
          drawDataPoints(data, 1);
          
          // Draw network architecture up to current layer
          drawNetworkArchitecture(network, progress, layerIdx);
          
          // Draw layer info
          ctx.fillStyle = COLORS.text;
          ctx.font = '14px Arial';
          ctx.textAlign = 'left';
          ctx.fillText(`Building Layer ${layerIdx + 1}/${params.layer_sizes.length}`, 60, 30);
          ctx.fillText(`Nodes: ${params.layer_sizes[layerIdx]}`, 60, 50);
        }
      }, { delay: layerIdx * 500 });
    }
    
    // Phase 3: Animate training process
    let currentEpoch = 0;
    
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw grid and points
        drawGrid();
        drawDataPoints(data, 1);
        
        // Train network in batches
        const targetEpoch = Math.floor(progress * epochs);
        if (targetEpoch > currentEpoch) {
          const epochsToTrain = Math.min(batchSize, targetEpoch - currentEpoch);
          network.train(data, epochsToTrain);
          currentEpoch += epochsToTrain;
        }
        
        // Draw network architecture with activations
        drawNetworkArchitecture(network, 1);
        
        // Draw decision boundary if enabled
        if (params.show_activations) {
          drawDecisionBoundary(network, progress);
        }
        
        // Draw loss curve if enabled
        if (params.show_gradients) {
          drawLossCurve(network.lossHistory, progress);
        }
        
        // Draw training info
        ctx.fillStyle = COLORS.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Epoch: ${currentEpoch}/${epochs}`, 60, 30);
        ctx.fillText(`Learning Rate: ${params.learning_rate}`, 60, 50);
        ctx.fillText(`Activation: ${params.activation}`, 60, 70);
        ctx.fillText(`Loss: ${network.lossHistory.length > 0 ? network.lossHistory[network.lossHistory.length - 1].toFixed(4) : 'N/A'}`, 60, 90);
      }
    });
    
    // Phase 4: Final reveal with all elements
    timeline.add({
      duration: 2000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawGrid();
        drawDataPoints(data, 1);
        drawNetworkArchitecture(network, 1);
        drawDecisionBoundary(network, 1);
        drawLossCurve(network.lossHistory, 1);
        
        // Draw final stats
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Final Architecture: ${params.layer_sizes.join(' → ')}`, 60, 30);
        ctx.fillText(`Final Loss: ${network.lossHistory[network.lossHistory.length - 1].toFixed(4)}`, 60, 50);
        ctx.fillText(`Training Accuracy: ${calculateAccuracy(network, data).toFixed(2)}%`, 60, 70);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Helper function to calculate accuracy
  const calculateAccuracy = (network, data) => {
    let correct = 0;
    data.forEach(point => {
      if (network.predict(point) == point.label) {
        correct++;
      }
    });
    return (correct / data.length) * 100;
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'architecture':
      params.show_activations = false;
      params.show_gradients = false;
      animateNeuralNetwork();
      break;
      
    case 'training-process':
      params.show_activations = false;
      params.show_gradients = true;
      animateNeuralNetwork();
      break;
      
    case 'decision-boundary':
      params.show_activations = true;
      params.show_gradients = false;
      animateNeuralNetwork();
      break;
      
    case 'feature-visualization':
      params.show_activations = true;
      params.show_gradients = true;
      animateNeuralNetwork();
      break;
      
    case 'all':
      params.show_activations = true;
      params.show_gradients = true;
      animateNeuralNetwork();
      break;
      
    default:
      animateNeuralNetwork();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Number of Layers',
        type: 'range',
        min: 2,
        max: 5,
        step: 1,
        value: params.n_layers,
        onChange: (value) => {
          params.n_layers = parseInt(value);
          // Update layer sizes based on number of layers
          const newLayerSizes = [2]; // Input layer always has 2 nodes
          for (let i = 1; i < params.n_layers - 1; i++) {
            newLayerSizes.push(8); // Hidden layers
          }
          newLayerSizes.push(params.layer_sizes[params.layer_sizes.length - 1]); // Output layer
          params.layer_sizes = newLayerSizes;
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Learning Rate',
        type: 'range',
        min: 0.001,
        max: 0.1,
        step: 0.001,
        value: params.learning_rate,
        onChange: (value) => {
          params.learning_rate = parseFloat(value);
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Activation Function',
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
        label: 'Data Distribution',
        type: 'select',
        options: [
          { value: 'spiral', label: 'Spiral', selected: params.data_distribution === 'spiral' },
          { value: 'concentric', label: 'Concentric', selected: params.data_distribution === 'concentric' },
          { value: 'linear', label: 'Linear', selected: params.data_distribution === 'linear' }
        ],
        onChange: (value) => {
          params.data_distribution = value;
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Activations',
        type: 'checkbox',
        checked: params.show_activations,
        onChange: (value) => {
          params.show_activations = value;
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Gradients',
        type: 'checkbox',
        checked: params.show_gradients,
        onChange: (value) => {
          params.show_gradients = value;
          visualizeNeuralNetwork(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'architecture', label: 'Architecture', selected: visualizationType === 'architecture' },
          { value: 'training-process', label: 'Training Process', selected: visualizationType === 'training-process' },
          { value: 'decision-boundary', label: 'Decision Boundary', selected: visualizationType === 'decision-boundary' },
          { value: 'feature-visualization', label: 'Feature Visualization', selected: visualizationType === 'feature-visualization' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
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
// Enhanced Convolutional Neural Network Visualizations
// =============================================
function visualizeCNN(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    network_depth: 4,
    filter_size: 3,
    num_filters: 32,
    stride: 1,
    padding: 'same',
    pooling_type: 'max',
    activation: 'relu',
    input_image: 'mnist-digit',
    show_feature_maps: true,
    show_architecture: false,
    show_filters: false,
    show_training: false,
    animation_duration: 2500,
    interactive: true,
    controlsContainer: null
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 1000;
  const height = 600;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // CNN Model
  class CNN {
    constructor(depth, filterSize, numFilters, stride, padding, poolingType, activation) {
      this.depth = depth;
      this.filterSize = filterSize;
      this.numFilters = numFilters;
      this.stride = stride;
      this.padding = padding;
      this.poolingType = poolingType;
      this.activation = activation;
      this.filters = [];
      this.featureMaps = [];
      this.lossHistory = [];
      
      // Initialize filters
      for (let i = 0; i < depth; i++) {
        const layerFilters = [];
        for (let j = 0; j < numFilters; j++) {
          const filter = [];
          for (let k = 0; k < filterSize; k++) {
            const row = [];
            for (let l = 0; l < filterSize; l++) {
              row.push(MathUtils.random(-1, 1));
            }
            filter.push(row);
          }
          layerFilters.push(filter);
        }
        this.filters.push(layerFilters);
      }
    }
    
    activate(x) {
      switch (this.activation) {
        case 'sigmoid':
          return MathUtils.sigmoid(x);
        case 'tanh':
          return Math.tanh(x);
        case 'relu':
          return Math.max(0, x);
        default:
          return Math.max(0, x); // Default to ReLU
      }
    }
    
    convolve(input, filter) {
      const inputSize = input.length;
      const filterSize = filter.length;
      const outputSize = Math.floor((inputSize - filterSize) / this.stride) + 1;
      const output = Array(outputSize).fill(0).map(() => Array(outputSize).fill(0));
      
      for (let i = 0; i < outputSize; i++) {
        for (let j = 0; j < outputSize; j++) {
          let sum = 0;
          for (let k = 0; k < filterSize; k++) {
            for (let l = 0; l < filterSize; l++) {
              const inputRow = i * this.stride + k;
              const inputCol = j * this.stride + l;
              sum += input[inputRow][inputCol] * filter[k][l];
            }
          }
          output[i][j] = this.activate(sum);
        }
      }
      
      return output;
    }
    
    pool(input, poolSize = 2, poolType = 'max') {
      const inputSize = input.length;
      const outputSize = Math.floor(inputSize / poolSize);
      const output = Array(outputSize).fill(0).map(() => Array(outputSize).fill(0));
      
      for (let i = 0; i < outputSize; i++) {
        for (let j = 0; j < outputSize; j++) {
          const startRow = i * poolSize;
          const startCol = j * poolSize;
          let values = [];
          
          for (let k = 0; k < poolSize; k++) {
            for (let l = 0; l < poolSize; l++) {
              values.push(input[startRow + k][startCol + l]);
            }
          }
          
          if (poolType === 'max') {
            output[i][j] = Math.max(...values);
          } else if (poolType === 'avg') {
            output[i][j] = values.reduce((a, b) => a + b, 0) / values.length;
          }
        }
      }
      
      return output;
    }
    
    forwardPass(input) {
      this.featureMaps = [input];
      let currentOutput = input;
      
      for (let layer = 0; layer < this.depth; layer++) {
        const layerMaps = [];
        
        for (let filterIdx = 0; filterIdx < this.numFilters; filterIdx++) {
          const convolved = this.convolve(currentOutput, this.filters[layer][filterIdx]);
          layerMaps.push(convolved);
        }
        
        // Apply pooling
        if (layer < this.depth - 1) { // Don't pool the last layer
          const pooledMaps = layerMaps.map(map => this.pool(map, 2, this.poolingType));
          this.featureMaps.push(pooledMaps);
          currentOutput = pooledMaps[0]; // Use first feature map for next layer
        } else {
          this.featureMaps.push(layerMaps);
        }
      }
      
      return this.featureMaps;
    }
    
    // Simplified training for visualization
    train(input, target, epochs) {
      for (let epoch = 0; epoch < epochs; epoch++) {
        this.forwardPass(input);
        
        // Simplified loss calculation
        const finalMaps = this.featureMaps[this.featureMaps.length - 1];
        let loss = 0;
        
        for (let i = 0; i < finalMaps.length; i++) {
          for (let j = 0; j < finalMaps[i].length; j++) {
            for (let k = 0; k < finalMaps[i][j].length; k++) {
              // Simplified target: make feature maps more "distinct"
              const targetValue = (i + j + k) % 2 === 0 ? 1 : -1;
              loss += Math.pow(finalMaps[i][j][k] - targetValue, 2);
            }
          }
        }
        
        this.lossHistory.push(loss);
        
        // Simplified weight update
        for (let layer = 0; layer < this.depth; layer++) {
          for (let filterIdx = 0; filterIdx < this.numFilters; filterIdx++) {
            for (let i = 0; i < this.filterSize; i++) {
              for (let j = 0; j < this.filterSize; j++) {
                // Simplified gradient
                const gradient = MathUtils.random(-0.1, 0.1);
                this.filters[layer][filterIdx][i][j] -= 0.01 * gradient;
              }
            }
          }
        }
      }
    }
  }
  
  // Generate sample input image
  const generateInputImage = (type = 'mnist-digit', size = 28) => {
    const image = Array(size).fill(0).map(() => Array(size).fill(0));
    
    switch (type) {
      case 'mnist-digit':
        // Draw a simple digit-like shape
        const center = size / 2;
        const radius = size / 4;
        
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            const distance = Math.sqrt(Math.pow(i - center, 2) + Math.pow(j - center, 2));
            const angle = Math.atan2(i - center, j - center);
            
            // Create a digit-like pattern
            if (distance < radius && distance > radius * 0.6) {
              // Vertical stroke
              if (Math.abs(angle) < Math.PI / 4 || Math.abs(angle) > 3 * Math.PI / 4) {
                image[i][j] = 1;
              }
              
              // Horizontal stroke
              if (Math.abs(angle - Math.PI / 2) < Math.PI / 6) {
                image[i][j] = 1;
              }
            }
            
            // Add some noise
            image[i][j] += MathUtils.random(0, 0.1);
            image[i][j] = Math.min(1, Math.max(0, image[i][j]));
          }
        }
        break;
        
      case 'checkerboard':
        // Checkerboard pattern
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            image[i][j] = (Math.floor(i / 4) + Math.floor(j / 4)) % 2;
          }
        }
        break;
        
      case 'vertical-lines':
        // Vertical lines
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            image[i][j] = j % 8 < 4 ? 1 : 0;
          }
        }
        break;
        
      case 'horizontal-lines':
        // Horizontal lines
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            image[i][j] = i % 8 < 4 ? 1 : 0;
          }
        }
        break;
        
      default:
        // Random pattern
        for (let i = 0; i < size; i++) {
          for (let j = 0; j < size; j++) {
            image[i][j] = MathUtils.random(0, 1);
          }
        }
    }
    
    return image;
  };
  
  // Draw input image
  const drawInputImage = (image, x, y, width, height, title = 'Input Image') => {
    ctx.save();
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(title, x + width / 2, y - 5);
    
    // Draw image
    const cellWidth = width / image.length;
    const cellHeight = height / image[0].length;
    
    for (let i = 0; i < image.length; i++) {
      for (let j = 0; j < image[i].length; j++) {
        const intensity = Math.floor(image[i][j] * 255);
        ctx.fillStyle = `rgb(${intensity}, ${intensity}, ${intensity})`;
        ctx.fillRect(
          x + i * cellWidth,
          y + j * cellHeight,
          cellWidth,
          cellHeight
        );
      }
    }
    
    ctx.restore();
  };
  
  // Draw feature maps
  const drawFeatureMaps = (featureMaps, x, y, width, height, layer, progress = 1) => {
    ctx.save();
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`Layer ${layer + 1} Feature Maps`, x + width / 2, y - 5);
    
    // Draw feature maps
    const maps = featureMaps[layer];
    const mapsPerRow = Math.ceil(Math.sqrt(maps.length));
    const mapWidth = width / mapsPerRow;
    const mapHeight = height / mapsPerRow;
    
    for (let mapIdx = 0; mapIdx < maps.length; mapIdx++) {
      const row = Math.floor(mapIdx / mapsPerRow);
      const col = mapIdx % mapsPerRow;
      const mapX = x + col * mapWidth;
      const mapY = y + row * mapHeight;
      
      // Only draw if this map is within the current progress
      if (mapIdx < maps.length * progress) {
        const map = maps[mapIdx];
        const cellWidth = mapWidth / map.length;
        const cellHeight = mapHeight / map[0].length;
        
        for (let i = 0; i < map.length; i++) {
          for (let j = 0; j < map[i].length; j++) {
            // Use color to represent positive/negative activations
            const value = map[i][j];
            let r, g, b;
            
            if (value > 0) {
              // Positive activations in blue
              const intensity = Math.floor(Math.min(1, value) * 255);
              r = 0;
              g = 0;
              b = intensity;
            } else {
              // Negative activations in red
              const intensity = Math.floor(Math.min(1, -value) * 255);
              r = intensity;
              g = 0;
              b = 0;
            }
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(
              mapX + i * cellWidth,
              mapY + j * cellHeight,
              cellWidth,
              cellHeight
            );
          }
        }
        
        // Draw map border
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1;
        ctx.strokeRect(mapX, mapY, mapWidth, mapHeight);
        
        // Draw filter number
        ctx.fillStyle = COLORS.text;
        ctx.font = '8px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Filter ${mapIdx + 1}`, mapX + mapWidth / 2, mapY + mapHeight + 10);
      }
    }
    
    ctx.restore();
  };
  
  // Draw filters
  const drawFilters = (filters, x, y, width, height, layer, progress = 1) => {
    ctx.save();
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`Layer ${layer + 1} Filters`, x + width / 2, y - 5);
    
    // Draw filters
    const layerFilters = filters[layer];
    const filtersPerRow = Math.ceil(Math.sqrt(layerFilters.length));
    const filterWidth = width / filtersPerRow;
    const filterHeight = height / filtersPerRow;
    
    for (let filterIdx = 0; filterIdx < layerFilters.length; filterIdx++) {
      const row = Math.floor(filterIdx / filtersPerRow);
      const col = filterIdx % filtersPerRow;
      const filterX = x + col * filterWidth;
      const filterY = y + row * filterHeight;
      
      // Only draw if this filter is within the current progress
      if (filterIdx < layerFilters.length * progress) {
        const filter = layerFilters[filterIdx];
        const cellWidth = filterWidth / filter.length;
        const cellHeight = filterHeight / filter[0].length;
        
        for (let i = 0; i < filter.length; i++) {
          for (let j = 0; j < filter[i].length; j++) {
            // Use color to represent positive/negative weights
            const value = filter[i][j];
            let r, g, b;
            
            if (value > 0) {
              // Positive weights in green
              const intensity = Math.floor(Math.min(1, value) * 255);
              r = 0;
              g = intensity;
              b = 0;
            } else {
              // Negative weights in red
              const intensity = Math.floor(Math.min(1, -value) * 255);
              r = intensity;
              g = 0;
              b = 0;
            }
            
            ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
            ctx.fillRect(
              filterX + i * cellWidth,
              filterY + j * cellHeight,
              cellWidth,
              cellHeight
            );
          }
        }
        
        // Draw filter border
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1;
        ctx.strokeRect(filterX, filterY, filterWidth, filterHeight);
        
        // Draw filter number
        ctx.fillStyle = COLORS.text;
        ctx.font = '8px Arial';
        ctx.textAlign = 'center';
        ctx.fillText(`Filter ${filterIdx + 1}`, filterX + filterWidth / 2, filterY + filterHeight + 10);
      }
    }
    
    ctx.restore();
  };
  
  // Draw CNN architecture
  const drawArchitecture = (cnn, x, y, width, height, progress = 1) => {
    ctx.save();
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('CNN Architecture', x + width / 2, y - 5);
    
    // Draw layers
    const layerWidth = width / (cnn.depth + 2); // +2 for input and output
    const centerY = y + height / 2;
    
    // Draw input layer
    ctx.fillStyle = COLORS.accent;
    ctx.beginPath();
    ctx.arc(x + layerWidth / 2, centerY, 20, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = COLORS.text;
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Input', x + layerWidth / 2, centerY);
    
    // Draw convolutional layers
    for (let i = 0; i < cnn.depth; i++) {
      const layerX = x + (i + 1) * layerWidth + layerWidth / 2;
      const layerProgress = MathUtils.clamp((progress - i * 0.2) * 1.25, 0, 1);
      
      if (layerProgress > 0) {
        ctx.fillStyle = COLORS.spectrum[i % COLORS.spectrum.length];
        ctx.beginPath();
        ctx.arc(layerX, centerY, 20, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1;
        ctx.stroke();
        
        ctx.fillStyle = COLORS.text;
        ctx.font = '10px Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(`Conv ${i + 1}`, layerX, centerY);
        
        // Draw connections
        if (i > 0) {
          const prevX = x + i * layerWidth + layerWidth / 2;
          ctx.beginPath();
          ctx.moveTo(prevX + 20, centerY);
          ctx.lineTo(layerX - 20, centerY);
          ctx.strokeStyle = COLORS.text;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
    
    // Draw output layer
    const outputX = x + (cnn.depth + 1) * layerWidth + layerWidth / 2;
    ctx.fillStyle = COLORS.accent;
    ctx.beginPath();
    ctx.arc(outputX, centerY, 20, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = COLORS.text;
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Output', outputX, centerY);
    
    // Draw connection to output
    const lastConvX = x + cnn.depth * layerWidth + layerWidth / 2;
    ctx.beginPath();
    ctx.moveTo(lastConvX + 20, centerY);
    ctx.lineTo(outputX - 20, centerY);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.stroke();
    
    ctx.restore();
  };
  
  // Draw loss curve
  const drawLossCurve = (lossHistory, x, y, width, height, progress = 1) => {
    if (lossHistory.length === 0) return;
    
    ctx.save();
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Loss Curve', x + width / 2, y - 5);
    
    // Find min and max loss for scaling
    const maxLoss = Math.max(...lossHistory);
    const minLoss = Math.min(...lossHistory);
    const lossRange = maxLoss - minLoss || 1; // Avoid division by zero
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + 30, y + 20);
    ctx.lineTo(x + 30, y + height - 20);
    ctx.lineTo(x + width - 20, y + height - 20);
    ctx.stroke();
    
    // Draw labels
    ctx.textAlign = 'right';
    ctx.fillText(minLoss.toFixed(2), x + 25, y + height - 20);
    ctx.fillText(maxLoss.toFixed(2), x + 25, y + 20);
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', x + width / 2, y + height);
    
    // Draw loss curve
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const pointsToShow = Math.floor(lossHistory.length * progress);
    const visibleLosses = lossHistory.slice(0, pointsToShow);
    
    visibleLosses.forEach((loss, i) => {
      const pointX = x + 30 + (width - 50) * (i / (lossHistory.length - 1));
      const pointY = y + height - 20 - (height - 40) * ((loss - minLoss) / lossRange);
      
      if (i === 0) {
        ctx.moveTo(pointX, pointY);
      } else {
        ctx.lineTo(pointX, pointY);
      }
    });
    
    ctx.stroke();
    
    // Draw current point
    if (pointsToShow > 0) {
      const currentLoss = lossHistory[pointsToShow - 1];
      const pointX = x + 30 + (width - 50) * ((pointsToShow - 1) / (lossHistory.length - 1));
      const pointY = y + height - 20 - (height - 40) * ((currentLoss - minLoss) / lossRange);
      
      ctx.fillStyle = COLORS.accent;
      ctx.beginPath();
      ctx.arc(pointX, pointY, 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw current loss value
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(currentLoss.toFixed(4), pointX + 5, pointY);
    }
    
    ctx.restore();
  };
  
  // Animate the CNN with enhanced cinematic effects
  const animateCNN = () => {
    const cnn = new CNN(
      params.network_depth,
      params.filter_size,
      params.num_filters,
      params.stride,
      params.padding,
      params.pooling_type,
      params.activation
    );
    
    const inputImage = generateInputImage(params.input_image, 28);
    const epochs = 20;
    let currentEpoch = 0;
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Show input image
    timeline.add({
      duration: 1000,
      easing: 'easeOutBack',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        drawInputImage(inputImage, 50, 50, 200, 200, 'Input Image', progress);
        
        // Draw explanation text
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Input Image Processing', width / 2, 30);
        
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('The CNN receives an input image and processes it through multiple layers', 300, 70);
        ctx.fillText('to extract hierarchical features for classification.', 300, 90);
      }
    });
    
    // Phase 2: Show filters for each layer
    for (let layer = 0; layer < params.network_depth; layer++) {
      timeline.add({
        duration: 1000,
        easing: 'easeOutBack',
        onUpdate: (progress) => {
          ctx.clearRect(0, 0, width, height);
          drawInputImage(inputImage, 50, 50, 200, 200);
          drawFilters(cnn.filters, 300, 50, 300, 200, layer, progress);
          
          // Draw explanation text
          ctx.fillStyle = COLORS.text;
          ctx.font = '16px Arial';
          ctx.textAlign = 'center';
          ctx.fillText(`Layer ${layer + 1} Filters`, width / 2, 30);
          
          ctx.font = '14px Arial';
          ctx.textAlign = 'left';
          ctx.fillText(`Each filter in layer ${layer + 1} detects different patterns in the input`, 50, 270);
          ctx.fillText(`Positive weights (green) and negative weights (red) create feature detectors.`, 50, 290);
        }
      }, { delay: layer * 500 });
    }
    
    // Phase 3: Show forward pass with feature maps
    timeline.add({
      duration: 2000,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Perform forward pass
        cnn.forwardPass(inputImage);
        
        // Draw input image
        drawInputImage(inputImage, 50, 50, 200, 200);
        
        // Draw feature maps for each layer
        const mapsPerRow = 2;
        const mapWidth = 200;
        const mapHeight = 200;
        
        for (let layer = 0; layer < cnn.depth; layer++) {
          const row = Math.floor(layer / mapsPerRow);
          const col = layer % mapsPerRow;
          const x = 300 + col * (mapWidth + 20);
          const y = 50 + row * (mapHeight + 50);
          
          // Only show layers that are within the current progress
          const layerProgress = MathUtils.clamp((progress - layer * 0.2) * 1.25, 0, 1);
          if (layerProgress > 0) {
            drawFeatureMaps(cnn.featureMaps, x, y, mapWidth, mapHeight, layer, layerProgress);
          }
        }
        
        // Draw explanation text
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Feature Extraction Process', width / 2, 30);
        
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('Each layer extracts increasingly complex features from the input image:', 50, height - 80);
        ctx.fillText('• Early layers detect edges and simple patterns', 50, height - 60);
        ctx.fillText('• Later layers combine these to detect complex shapes and objects', 50, height - 40);
      }
    });
    
    // Phase 4: Show training process
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Train network in batches
        const targetEpoch = Math.floor(progress * epochs);
        if (targetEpoch > currentEpoch) {
          cnn.train(inputImage, null, 1);
          currentEpoch++;
        }
        
        // Draw input image
        drawInputImage(inputImage, 50, 50, 200, 200);
        
        // Draw feature maps for the first layer
        drawFeatureMaps(cnn.featureMaps, 300, 50, 300, 200, 0, 1);
        
        // Draw filters for the first layer
        drawFilters(cnn.filters, 650, 50, 300, 200, 0, 1);
        
        // Draw loss curve if enabled
        if (params.show_training) {
          drawLossCurve(cnn.lossHistory, 50, 300, 300, 150, progress);
        }
        
        // Draw architecture if enabled
        if (params.show_architecture) {
          drawArchitecture(cnn, 400, 300, 400, 150, progress);
        }
        
        // Draw training info
        ctx.fillStyle = COLORS.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Epoch: ${currentEpoch}/${epochs}`, 50, height - 80);
        ctx.fillText(`Learning Rate: 0.01`, 50, height - 60);
        ctx.fillText(`Loss: ${cnn.lossHistory.length > 0 ? cnn.lossHistory[cnn.lossHistory.length - 1].toFixed(4) : 'N/A'}`, 50, height - 40);
      }
    });
    
    // Phase 5: Final reveal with all elements
    timeline.add({
      duration: 2000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawInputImage(inputImage, 50, 50, 200, 200);
        
        // Draw feature maps for all layers
        const mapsPerRow = 2;
        const mapWidth = 200;
        const mapHeight = 200;
        
        for (let layer = 0; layer < cnn.depth; layer++) {
          const row = Math.floor(layer / mapsPerRow);
          const col = layer % mapsPerRow;
          const x = 300 + col * (mapWidth + 20);
          const y = 50 + row * (mapHeight + 50);
          
          drawFeatureMaps(cnn.featureMaps, x, y, mapWidth, mapHeight, layer, 1);
        }
        
        // Draw filters for the first layer
        drawFilters(cnn.filters, 50, 300, 200, 150, 0, 1);
        
        // Draw loss curve
        drawLossCurve(cnn.lossHistory, 300, 300, 300, 150, 1);
        
        // Draw architecture
        drawArchitecture(cnn, 650, 300, 300, 150, 1);
        
        // Draw final stats
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Final Architecture: ${params.network_depth} layers`, 50, height - 80);
        ctx.fillText(`Filters per Layer: ${params.num_filters}`, 50, height - 60);
        ctx.fillText(`Final Loss: ${cnn.lossHistory[cnn.lossHistory.length - 1].toFixed(4)}`, 50, height - 40);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'feature-maps':
      params.show_architecture = false;
      params.show_filters = false;
      params.show_training = false;
      animateCNN();
      break;
      
    case 'architecture':
      params.show_feature_maps = false;
      params.show_filters = false;
      params.show_training = false;
      animateCNN();
      break;
      
    case 'filter-visualization':
      params.show_feature_maps = false;
      params.show_architecture = false;
      params.show_training = false;
      animateCNN();
      break;
      
    case 'training-process':
      params.show_feature_maps = true;
      params.show_architecture = true;
      params.show_filters = true;
      params.show_training = true;
      animateCNN();
      break;
      
    case 'all':
      params.show_feature_maps = true;
      params.show_architecture = true;
      params.show_filters = true;
      params.show_training = true;
      animateCNN();
      break;
      
    default:
      animateCNN();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'Network Depth',
        type: 'range',
        min: 2,
        max: 6,
        step: 1,
        value: params.network_depth,
        onChange: (value) => {
          params.network_depth = parseInt(value);
          visualizeCNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Number of Filters',
        type: 'range',
        min: 8,
        max: 64,
        step: 8,
        value: params.num_filters,
        onChange: (value) => {
          params.num_filters = parseInt(value);
          visualizeCNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Filter Size',
        type: 'range',
        min: 3,
        max: 7,
        step: 2,
        value: params.filter_size,
        onChange: (value) => {
          params.filter_size = parseInt(value);
          visualizeCNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Input Image',
        type: 'select',
        options: [
          { value: 'mnist-digit', label: 'MNIST Digit', selected: params.input_image === 'mnist-digit' },
          { value: 'checkerboard', label: 'Checkerboard', selected: params.input_image === 'checkerboard' },
          { value: 'vertical-lines', label: 'Vertical Lines', selected: params.input_image === 'vertical-lines' },
          { value: 'horizontal-lines', label: 'Horizontal Lines', selected: params.input_image === 'horizontal-lines' },
          { value: 'random', label: 'Random', selected: params.input_image === 'random' }
        ],
        onChange: (value) => {
          params.input_image = value;
          visualizeCNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Feature Maps',
        type: 'checkbox',
        checked: params.show_feature_maps,
        onChange: (value) => {
          params.show_feature_maps = value;
          visualizeCNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Architecture',
        type: 'checkbox',
        checked: params.show_architecture,
        onChange: (value) => {
          params.show_architecture = value;
          visualizeCNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Filters',
        type: 'checkbox',
        checked: params.show_filters,
        onChange: (value) => {
          params.show_filters = value;
          visualizeCNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Training',
        type: 'checkbox',
        checked: params.show_training,
        onChange: (value) => {
          params.show_training = value;
          visualizeCNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'feature-maps', label: 'Feature Maps', selected: visualizationType === 'feature-maps' },
          { value: 'architecture', label: 'Architecture', selected: visualizationType === 'architecture' },
          { value: 'filter-visualization', label: 'Filter Visualization', selected: visualizationType === 'filter-visualization' },
          { value: 'training-process', label: 'Training Process', selected: visualizationType === 'training-process' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizeCNN(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'CNN Parameters',
      description: 'Adjust parameters to see how they affect the convolutional neural network.'
    });
  }
}

// =============================================
// Enhanced Recurrent Neural Network Visualizations
// =============================================
function visualizeRNN(containerId, visualizationType, params = {}) {
  // Default parameters with more options
  const defaultParams = {
    sequence_length: 10,
    hidden_size: 32,
    rnn_type: 'lstm',
    input_size: 8,
    output_size: 1,
    show_unrolled: true,
    show_rolled: false,
    show_hidden_states: false,
    show_gates: false,
    animation_duration: 3000,
    interactive: true,
    controlsContainer: null
  };
  
  // Merge with provided params
  params = { ...defaultParams, ...params };
  
  // Create canvas with enhanced options
  const width = 1000;
  const height = 600;
  const { canvas, ctx } = DomUtils.createCanvas(containerId, width, height, {
    background: '#ffffff',
    borderRadius: '8px',
    boxShadow: '0 4px 12px rgba(0, 0, 0, 0.1)'
  });
  if (!canvas) return;
  
  // RNN Model
  class RNN {
    constructor(inputSize, hiddenSize, outputSize, rnnType = 'simple') {
      this.inputSize = inputSize;
      this.hiddenSize = hiddenSize;
      this.outputSize = outputSize;
      this.rnnType = rnnType;
      
      // Initialize weights
      this.Wxh = this.initializeWeights(inputSize, hiddenSize); // Input to hidden
      this.Whh = this.initializeWeights(hiddenSize, hiddenSize); // Hidden to hidden
      this.Why = this.initializeWeights(hiddenSize, outputSize); // Hidden to output
      this.bh = new Array(hiddenSize).fill(0.1); // Hidden bias
      this.by = new Array(outputSize).fill(0.1); // Output bias
      
      // LSTM/GRU specific weights
      if (rnnType === 'lstm' || rnnType === 'gru') {
        // Gate weights
        this.Wxi = this.initializeWeights(inputSize, hiddenSize);
        this.Whi = this.initializeWeights(hiddenSize, hiddenSize);
        this.bi = new Array(hiddenSize).fill(0.1);
        
        this.Wxf = this.initializeWeights(inputSize, hiddenSize);
        this.Whf = this.initializeWeights(hiddenSize, hiddenSize);
        this.bf = new Array(hiddenSize).fill(0.1);
        
        this.Wxo = this.initializeWeights(inputSize, hiddenSize);
        this.Who = this.initializeWeights(hiddenSize, hiddenSize);
        this.bo = new Array(hiddenSize).fill(0.1);
        
        // Cell state weights (LSTM only)
        if (rnnType === 'lstm') {
          this.Wxc = this.initializeWeights(inputSize, hiddenSize);
          this.Whc = this.initializeWeights(hiddenSize, hiddenSize);
          this.bc = new Array(hiddenSize).fill(0.1);
        }
      }
      
      this.hiddenStates = [];
      this.outputs = [];
      this.gateStates = [];
      this.lossHistory = [];
    }
    
    initializeWeights(rows, cols) {
      const weights = [];
      for (let i = 0; i < rows; i++) {
        weights.push(new Array(cols).fill(0).map(() => MathUtils.random(-0.1, 0.1)));
      }
      return weights;
    }
    
    sigmoid(x) {
      return 1 / (1 + Math.exp(-x));
    }
    
    tanh(x) {
      return Math.tanh(x);
    }
    
    forwardStep(input, prevHidden) {
      const hidden = new Array(this.hiddenSize).fill(0);
      const gates = {};
      
      if (this.rnnType === 'simple') {
        // Simple RNN
        for (let i = 0; i < this.hiddenSize; i++) {
          let sum = this.bh[i];
          
          // Input to hidden
          for (let j = 0; j < this.inputSize; j++) {
            sum += input[j] * this.Wxh[j][i];
          }
          
          // Hidden to hidden
          for (let j = 0; j < this.hiddenSize; j++) {
            sum += prevHidden[j] * this.Whh[j][i];
          }
          
          hidden[i] = Math.tanh(sum);
        }
      } else if (this.rnnType === 'lstm') {
        // LSTM
        const inputGate = new Array(this.hiddenSize).fill(0);
        const forgetGate = new Array(this.hiddenSize).fill(0);
        const outputGate = new Array(this.hiddenSize).fill(0);
        const cellState = new Array(this.hiddenSize).fill(0);
        const newCellState = new Array(this.hiddenSize).fill(0);
        
        for (let i = 0; i < this.hiddenSize; i++) {
          // Input gate
          let sum_i = this.bi[i];
          for (let j = 0; j < this.inputSize; j++) {
            sum_i += input[j] * this.Wxi[j][i];
          }
          for (let j = 0; j < this.hiddenSize; j++) {
            sum_i += prevHidden[j] * this.Whi[j][i];
          }
          inputGate[i] = this.sigmoid(sum_i);
          
          // Forget gate
          let sum_f = this.bf[i];
          for (let j = 0; j < this.inputSize; j++) {
            sum_f += input[j] * this.Wxf[j][i];
          }
          for (let j = 0; j < this.hiddenSize; j++) {
            sum_f += prevHidden[j] * this.Whf[j][i];
          }
          forgetGate[i] = this.sigmoid(sum_f);
          
          // Output gate
          let sum_o = this.bo[i];
          for (let j = 0; j < this.inputSize; j++) {
            sum_o += input[j] * this.Wxo[j][i];
          }
          for (let j = 0; j < this.hiddenSize; j++) {
            sum_o += prevHidden[j] * this.Who[j][i];
          }
          outputGate[i] = this.sigmoid(sum_o);
          
          // Cell state
          let sum_c = this.bc[i];
          for (let j = 0; j < this.inputSize; j++) {
            sum_c += input[j] * this.Wxc[j][i];
          }
          for (let j = 0; j < this.hiddenSize; j++) {
            sum_c += prevHidden[j] * this.Whc[j][i];
          }
          cellState[i] = this.tanh(sum_c);
          
          // Update cell state
          newCellState[i] = forgetGate[i] * (prevHidden[i] || 0) + inputGate[i] * cellState[i];
          
          // Hidden state
          hidden[i] = outputGate[i] * this.tanh(newCellState[i]);
        }
        
        gates.inputGate = inputGate;
        gates.forgetGate = forgetGate;
        gates.outputGate = outputGate;
        gates.cellState = newCellState;
      } else if (this.rnnType === 'gru') {
        // GRU
        const resetGate = new Array(this.hiddenSize).fill(0);
        const updateGate = new Array(this.hiddenSize).fill(0);
        const candidateState = new Array(this.hiddenSize).fill(0);
        
        for (let i = 0; i < this.hiddenSize; i++) {
          // Reset gate
          let sum_r = this.bi[i];
          for (let j = 0; j < this.inputSize; j++) {
            sum_r += input[j] * this.Wxi[j][i];
          }
          for (let j = 0; j < this.hiddenSize; j++) {
            sum_r += prevHidden[j] * this.Whi[j][i];
          }
          resetGate[i] = this.sigmoid(sum_r);
          
          // Update gate
          let sum_u = this.bf[i];
          for (let j = 0; j < this.inputSize; j++) {
            sum_u += input[j] * this.Wxf[j][i];
          }
          for (let j = 0; j < this.hiddenSize; j++) {
            sum_u += prevHidden[j] * this.Whf[j][i];
          }
          updateGate[i] = this.sigmoid(sum_u);
          
          // Candidate state
          let sum_c = this.bc[i];
          for (let j = 0; j < this.inputSize; j++) {
            sum_c += input[j] * this.Wxc[j][i];
          }
          for (let j = 0; j < this.hiddenSize; j++) {
            sum_c += (resetGate[j] * prevHidden[j]) * this.Whc[j][i];
          }
          candidateState[i] = this.tanh(sum_c);
          
          // Hidden state
          hidden[i] = (1 - updateGate[i]) * candidateState[i] + updateGate[i] * (prevHidden[i] || 0);
        }
        
        gates.resetGate = resetGate;
        gates.updateGate = updateGate;
        gates.candidateState = candidateState;
      }
      
      // Calculate output
      const output = new Array(this.outputSize).fill(0);
      for (let i = 0; i < this.outputSize; i++) {
        let sum = this.by[i];
        for (let j = 0; j < this.hiddenSize; j++) {
          sum += hidden[j] * this.Why[j][i];
        }
        output[i] = sum; // Linear output for regression
      }
      
      return { hidden, output, gates };
    }
    
    forwardPass(sequence) {
      this.hiddenStates = [];
      this.outputs = [];
      this.gateStates = [];
      
      let hidden = new Array(this.hiddenSize).fill(0);
      
      for (let t = 0; t < sequence.length; t++) {
        const result = this.forwardStep(sequence[t], hidden);
        hidden = result.hidden;
        
        this.hiddenStates.push([...hidden]);
        this.outputs.push([...result.output]);
        this.gateStates.push({ ...result.gates });
      }
      
      return this.outputs;
    }
    
    // Simplified training for visualization
    train(sequence, target, epochs) {
      for (let epoch = 0; epoch < epochs; epoch++) {
        this.forwardPass(sequence);
        
        // Simplified loss calculation
        let loss = 0;
        for (let t = 0; t < sequence.length; t++) {
          for (let i = 0; i < this.outputSize; i++) {
            loss += Math.pow(this.outputs[t][i] - target[t][i], 2);
          }
        }
        
        this.lossHistory.push(loss);
        
        // Simplified weight update
        for (let i = 0; i < this.inputSize; i++) {
          for (let j = 0; j < this.hiddenSize; j++) {
            this.Wxh[i][j] += MathUtils.random(-0.01, 0.01);
          }
        }
        
        for (let i = 0; i < this.hiddenSize; i++) {
          for (let j = 0; j < this.hiddenSize; j++) {
            this.Whh[i][j] += MathUtils.random(-0.01, 0.01);
          }
        }
      }
    }
  }
  
  // Generate sample sequence data
  const generateSequence = (length, inputSize) => {
    const sequence = [];
    const target = [];
    
    // Simple sine wave pattern
    for (let t = 0; t < length; t++) {
      const input = new Array(inputSize).fill(0);
      
      // Create a pattern with multiple frequencies
      for (let i = 0; i < inputSize; i++) {
        const frequency = 0.1 + 0.2 * i;
        input[i] = Math.sin(t * frequency) + MathUtils.random(-0.1, 0.1);
      }
      
      sequence.push(input);
      
      // Target is the next value of the first feature
      target.push([input[0]]);
    }
    
    return { sequence, target };
  };
  
  // Draw unrolled RNN
  const drawUnrolledRNN = (rnn, sequence, x, y, width, height, progress = 1) => {
    ctx.save();
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Unrolled RNN Across Time Steps', x + width / 2, y - 5);
    
    // Calculate positions
    const timeStepWidth = width / (sequence.length + 1);
    const centerY = y + height / 2;
    const nodeRadius = 15;
    
    // Draw input nodes
    for (let t = 0; t < sequence.length; t++) {
      const stepProgress = MathUtils.clamp((progress - t * 0.1) * 1.1, 0, 1);
      if (stepProgress <= 0) continue;
      
      const stepX = x + (t + 1) * timeStepWidth;
      
      // Draw input node
      ctx.fillStyle = COLORS.accent;
      ctx.globalAlpha = stepProgress;
      ctx.beginPath();
      ctx.arc(stepX, centerY - 80, nodeRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1;
      ctx.stroke();
      
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`X${t}`, stepX, centerY - 80);
      
      // Draw input values
      ctx.font = '8px Arial';
      for (let i = 0; i < Math.min(3, sequence[t].length); i++) {
        ctx.fillText(sequence[t][i].toFixed(2), stepX, centerY - 80 + 15 + i * 10);
      }
    }
    
    // Draw hidden nodes
    for (let t = 0; t < sequence.length; t++) {
      const stepProgress = MathUtils.clamp((progress - t * 0.1) * 1.1, 0, 1);
      if (stepProgress <= 0) continue;
      
      const stepX = x + (t + 1) * timeStepWidth;
      
      // Draw hidden node
      ctx.fillStyle = COLORS.spectrum[1];
      ctx.globalAlpha = stepProgress;
      ctx.beginPath();
      ctx.arc(stepX, centerY, nodeRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1;
      ctx.stroke();
      
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`H${t}`, stepX, centerY);
      
      // Draw hidden state values if available
      if (rnn.hiddenStates[t] && stepProgress > 0.5) {
        ctx.font = '8px Arial';
        for (let i = 0; i < Math.min(3, rnn.hiddenStates[t].length); i++) {
          ctx.fillText(rnn.hiddenStates[t][i].toFixed(2), stepX, centerY + 15 + i * 10);
        }
      }
    }
    
    // Draw output nodes
    for (let t = 0; t < sequence.length; t++) {
      const stepProgress = MathUtils.clamp((progress - t * 0.1) * 1.1, 0, 1);
      if (stepProgress <= 0) continue;
      
      const stepX = x + (t + 1) * timeStepWidth;
      
      // Draw output node
      ctx.fillStyle = COLORS.spectrum[2];
      ctx.globalAlpha = stepProgress;
      ctx.beginPath();
      ctx.arc(stepX, centerY + 80, nodeRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1;
      ctx.stroke();
      
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`Y${t}`, stepX, centerY + 80);
      
      // Draw output values if available
      if (rnn.outputs[t] && stepProgress > 0.5) {
        ctx.font = '8px Arial';
        for (let i = 0; i < rnn.outputs[t].length; i++) {
          ctx.fillText(rnn.outputs[t][i].toFixed(2), stepX, centerY + 80 + 15 + i * 10);
        }
      }
    }
    
    // Draw connections
    ctx.globalAlpha = 1;
    for (let t = 0; t < sequence.length; t++) {
      const stepProgress = MathUtils.clamp((progress - t * 0.1) * 1.1, 0, 1);
      if (stepProgress <= 0) continue;
      
      const stepX = x + (t + 1) * timeStepWidth;
      
      // Input to hidden
      ctx.beginPath();
      ctx.moveTo(stepX, centerY - 80 + nodeRadius);
      ctx.lineTo(stepX, centerY - nodeRadius);
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1;
      ctx.stroke();
      
      // Hidden to output
      ctx.beginPath();
      ctx.moveTo(stepX, centerY + nodeRadius);
      ctx.lineTo(stepX, centerY + 80 - nodeRadius);
      ctx.strokeStyle = COLORS.text;
      ctx.lineWidth = 1;
      ctx.stroke();
      
      // Hidden to hidden (recurrent)
      if (t > 0) {
        const prevX = x + t * timeStepWidth;
        ctx.beginPath();
        ctx.moveTo(prevX + nodeRadius, centerY);
        ctx.lineTo(stepX - nodeRadius, centerY);
        ctx.strokeStyle = COLORS.accent;
        ctx.lineWidth = 2;
        ctx.stroke();
        
        // Draw arrowhead
        ctx.beginPath();
        ctx.moveTo(stepX - nodeRadius, centerY);
        ctx.lineTo(stepX - nodeRadius - 5, centerY - 3);
        ctx.lineTo(stepX - nodeRadius - 5, centerY + 3);
        ctx.closePath();
        ctx.fillStyle = COLORS.accent;
        ctx.fill();
      }
    }
    
    ctx.restore();
  };
  
  // Draw rolled RNN
  const drawRolledRNN = (rnn, x, y, width, height, progress = 1) => {
    ctx.save();
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Rolled RNN with Recurrent Connection', x + width / 2, y - 5);
    
    // Calculate positions
    const centerX = x + width / 2;
    const centerY = y + height / 2;
    const nodeRadius = 20;
    
    // Draw input node
    ctx.fillStyle = COLORS.accent;
    ctx.beginPath();
    ctx.arc(centerX - 80, centerY, nodeRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Input', centerX - 80, centerY);
    
    // Draw hidden node
    ctx.fillStyle = COLORS.spectrum[1];
    ctx.beginPath();
    ctx.arc(centerX, centerY, nodeRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Hidden', centerX, centerY);
    
    // Draw output node
    ctx.fillStyle = COLORS.spectrum[2];
    ctx.beginPath();
    ctx.arc(centerX + 80, centerY, nodeRadius, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText('Output', centerX + 80, centerY);
    
    // Draw connections
    // Input to hidden
    ctx.beginPath();
    ctx.moveTo(centerX - 80 + nodeRadius, centerY);
    ctx.lineTo(centerX - nodeRadius, centerY);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Hidden to output
    ctx.beginPath();
    ctx.moveTo(centerX + nodeRadius, centerY);
    ctx.lineTo(centerX + 80 - nodeRadius, centerY);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Recurrent connection
    ctx.beginPath();
    ctx.arc(centerX, centerY, 50, Math.PI * 0.25, Math.PI * 1.75, false);
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.stroke();
    
    // Draw arrowhead
    const angle = Math.PI * 1.25;
    const arrowX = centerX + 50 * Math.cos(angle);
    const arrowY = centerY + 50 * Math.sin(angle);
    ctx.beginPath();
    ctx.moveTo(arrowX, arrowY);
    ctx.lineTo(arrowX - 8 * Math.cos(angle - Math.PI / 6), arrowY - 8 * Math.sin(angle - Math.PI / 6));
    ctx.lineTo(arrowX - 8 * Math.cos(angle + Math.PI / 6), arrowY - 8 * Math.sin(angle + Math.PI / 6));
    ctx.closePath();
    ctx.fillStyle = COLORS.accent;
    ctx.fill();
    
    // Draw RNN type
    ctx.fillStyle = COLORS.text;
    ctx.font = '14px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(rnn.rnnType.toUpperCase(), centerX, centerY - 60);
    
    ctx.restore();
  };
  
  // Draw hidden state evolution
  const drawHiddenStates = (rnn, x, y, width, height, progress = 1) => {
    if (!rnn.hiddenStates || rnn.hiddenStates.length === 0) return;
    
    ctx.save();
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Hidden State Evolution Over Time', x + width / 2, y - 5);
    
    // Find min and max values for scaling
    let minVal = Infinity;
    let maxVal = -Infinity;
    
    for (let t = 0; t < rnn.hiddenStates.length; t++) {
      for (let i = 0; i < rnn.hiddenStates[t].length; i++) {
        minVal = Math.min(minVal, rnn.hiddenStates[t][i]);
        maxVal = Math.max(maxVal, rnn.hiddenStates[t][i]);
      }
    }
    
    const valueRange = maxVal - minVal || 1;
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + 30, y + 20);
    ctx.lineTo(x + 30, y + height - 20);
    ctx.lineTo(x + width - 20, y + height - 20);
    ctx.stroke();
    
    // Draw labels
    ctx.textAlign = 'right';
    ctx.fillText(minVal.toFixed(2), x + 25, y + height - 20);
    ctx.fillText(maxVal.toFixed(2), x + 25, y + 20);
    ctx.textAlign = 'center';
    ctx.fillText('Time Step', x + width / 2, y + height);
    ctx.textAlign = 'right';
    ctx.fillText('Hidden State Value', x + 15, y + height / 2);
    
    // Draw hidden state trajectories
    const timeSteps = rnn.hiddenStates.length;
    const hiddenSize = rnn.hiddenStates[0].length;
    const colors = COLORS.spectrum;
    
    for (let i = 0; i < Math.min(5, hiddenSize); i++) {
      ctx.strokeStyle = colors[i % colors.length];
      ctx.lineWidth = 2;
      ctx.beginPath();
      
      for (let t = 0; t < timeSteps; t++) {
        const pointX = x + 30 + (width - 50) * (t / (timeSteps - 1));
        const pointY = y + height - 20 - (height - 40) * ((rnn.hiddenStates[t][i] - minVal) / valueRange);
        
        if (t === 0) {
          ctx.moveTo(pointX, pointY);
        } else {
          ctx.lineTo(pointX, pointY);
        }
      }
      
      ctx.stroke();
      
      // Draw current point
      const currentStep = Math.floor(timeSteps * progress) - 1;
      if (currentStep >= 0 && currentStep < timeSteps) {
        const pointX = x + 30 + (width - 50) * (currentStep / (timeSteps - 1));
        const pointY = y + height - 20 - (height - 40) * ((rnn.hiddenStates[currentStep][i] - minVal) / valueRange);
        
        ctx.fillStyle = colors[i % colors.length];
        ctx.beginPath();
        ctx.arc(pointX, pointY, 4, 0, Math.PI * 2);
        ctx.fill();
        
        // Draw label
        if (progress > 0.9) {
          ctx.fillStyle = COLORS.text;
          ctx.font = '10px Arial';
          ctx.textAlign = 'left';
          ctx.fillText(`H${i}`, pointX + 5, pointY);
        }
      }
    }
    
    ctx.restore();
  };
  
  // Draw gate operations
  const drawGateOperations = (rnn, x, y, width, height, timeStep, progress = 1) => {
    if (!rnn.gateStates || rnn.gateStates.length === 0) return;
    if (timeStep >= rnn.gateStates.length) return;
    
    ctx.save();
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(`Gate Operations at Time Step ${timeStep}`, x + width / 2, y - 5);
    
    const gates = rnn.gateStates[timeStep];
    const gateTypes = Object.keys(gates);
    const gateHeight = height / (gateTypes.length + 1);
    
    for (let i = 0; i < gateTypes.length; i++) {
      const gateType = gateTypes[i];
      const gateY = y + (i + 1) * gateHeight;
      const gateValues = gates[gateType];
      
      // Draw gate title
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(gateType, x + 10, gateY - 10);
      
      // Draw gate values
      const barWidth = (width - 100) / gateValues.length;
      
      for (let j = 0; j < gateValues.length; j++) {
        const value = gateValues[j];
        const barX = x + 80 + j * barWidth;
        const barHeight = Math.abs(value) * (gateHeight - 20);
        
        if (value > 0) {
          ctx.fillStyle = `rgba(76, 175, 80, ${Math.min(1, value)})`; // Green for positive
        } else {
          ctx.fillStyle = `rgba(244, 67, 54, ${Math.min(1, -value)})`; // Red for negative
        }
        
        ctx.fillRect(barX, gateY - barHeight / 2, barWidth - 2, barHeight);
        
        // Draw value label
        if (progress > 0.8 && j % 5 === 0) {
          ctx.fillStyle = COLORS.text;
          ctx.font = '8px Arial';
          ctx.textAlign = 'center';
          ctx.fillText(value.toFixed(2), barX + barWidth / 2, gateY + 10);
        }
      }
    }
    
    ctx.restore();
  };
  
  // Draw loss curve
  const drawLossCurve = (lossHistory, x, y, width, height, progress = 1) => {
    if (lossHistory.length === 0) return;
    
    ctx.save();
    
    // Draw background
    ctx.fillStyle = '#f8f9fa';
    ctx.fillRect(x, y, width, height);
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.strokeRect(x, y, width, height);
    
    // Draw title
    ctx.fillStyle = COLORS.text;
    ctx.font = '12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('Loss Curve', x + width / 2, y - 5);
    
    // Find min and max loss for scaling
    const maxLoss = Math.max(...lossHistory);
    const minLoss = Math.min(...lossHistory);
    const lossRange = maxLoss - minLoss || 1; // Avoid division by zero
    
    // Draw axes
    ctx.strokeStyle = COLORS.text;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x + 30, y + 20);
    ctx.lineTo(x + 30, y + height - 20);
    ctx.lineTo(x + width - 20, y + height - 20);
    ctx.stroke();
    
    // Draw labels
    ctx.textAlign = 'right';
    ctx.fillText(minLoss.toFixed(2), x + 25, y + height - 20);
    ctx.fillText(maxLoss.toFixed(2), x + 25, y + 20);
    ctx.textAlign = 'center';
    ctx.fillText('Epoch', x + width / 2, y + height);
    
    // Draw loss curve
    ctx.strokeStyle = COLORS.accent;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const pointsToShow = Math.floor(lossHistory.length * progress);
    const visibleLosses = lossHistory.slice(0, pointsToShow);
    
    visibleLosses.forEach((loss, i) => {
      const pointX = x + 30 + (width - 50) * (i / (lossHistory.length - 1));
      const pointY = y + height - 20 - (height - 40) * ((loss - minLoss) / lossRange);
      
      if (i === 0) {
        ctx.moveTo(pointX, pointY);
      } else {
        ctx.lineTo(pointX, pointY);
      }
    });
    
    ctx.stroke();
    
    // Draw current point
    if (pointsToShow > 0) {
      const currentLoss = lossHistory[pointsToShow - 1];
      const pointX = x + 30 + (width - 50) * ((pointsToShow - 1) / (lossHistory.length - 1));
      const pointY = y + height - 20 - (height - 40) * ((currentLoss - minLoss) / lossRange);
      
      ctx.fillStyle = COLORS.accent;
      ctx.beginPath();
      ctx.arc(pointX, pointY, 4, 0, Math.PI * 2);
      ctx.fill();
      
      // Draw current loss value
      ctx.fillStyle = COLORS.text;
      ctx.font = '10px Arial';
      ctx.textAlign = 'left';
      ctx.fillText(currentLoss.toFixed(4), pointX + 5, pointY);
    }
    
    ctx.restore();
  };
  
  // Animate the RNN with enhanced cinematic effects
  const animateRNN = () => {
    const { sequence, target } = generateSequence(params.sequence_length, params.input_size);
    const rnn = new RNN(params.input_size, params.hidden_size, params.output_size, params.rnn_type);
    const epochs = 20;
    let currentEpoch = 0;
    
    // Create timeline for animation
    const timeline = AnimationSystem.createTimeline();
    
    // Phase 1: Show input sequence
    timeline.add({
      duration: 1500,
      easing: 'easeOutBack',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw sequence
        const seqWidth = 800;
        const seqHeight = 100;
        const seqX = (width - seqWidth) / 2;
        const seqY = 50;
        
        ctx.fillStyle = '#f8f9fa';
        ctx.fillRect(seqX, seqY, seqWidth, seqHeight);
        ctx.strokeStyle = COLORS.text;
        ctx.lineWidth = 1;
        ctx.strokeRect(seqX, seqY, seqWidth, seqHeight);
        
        // Draw title
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Input Sequence', width / 2, seqY - 10);
        
        // Draw sequence values
        const stepWidth = seqWidth / sequence.length;
        
        for (let t = 0; t < sequence.length; t++) {
          const stepProgress = MathUtils.clamp((progress - t * 0.1) * 1.1, 0, 1);
          if (stepProgress <= 0) continue;
          
          const stepX = seqX + (t + 0.5) * stepWidth;
          
          ctx.fillStyle = COLORS.text;
          ctx.font = '12px Arial';
          ctx.textAlign = 'center';
          ctx.fillText(`t=${t}`, stepX, seqY + 20);
          
          // Draw value
          ctx.font = '10px Arial';
          for (let i = 0; i < Math.min(3, sequence[t].length); i++) {
            ctx.fillText(sequence[t][i].toFixed(2), stepX, seqY + 35 + i * 12);
          }
        }
        
        // Draw explanation
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('RNNs process sequential data one time step at a time,', 50, seqY + seqHeight + 50);
        ctx.fillText('maintaining a hidden state that captures information from previous steps.', 50, seqY + seqHeight + 70);
      }
    });
    
    // Phase 2: Show unrolled RNN
    timeline.add({
      duration: 2000,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Perform forward pass
        rnn.forwardPass(sequence);
        
        // Draw unrolled RNN
        drawUnrolledRNN(rnn, sequence, 50, 50, 900, 300, progress);
        
        // Draw explanation
        ctx.fillStyle = COLORS.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('The RNN is unrolled across time steps, with each step processing', 50, 400);
        ctx.fillText('one element of the sequence and updating the hidden state.', 50, 420);
      }
    });
    
    // Phase 3: Show rolled RNN
    timeline.add({
      duration: 1000,
      easing: 'easeOutBack',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw rolled RNN
        drawRolledRNN(rnn, 50, 50, 400, 200, progress);
        
        // Draw explanation
        ctx.fillStyle = COLORS.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText('The rolled view shows the compact representation with recurrent connections.', 50, 300);
        ctx.fillText(`This RNN uses ${params.rnn_type.toUpperCase()} cells for better memory retention.`, 50, 320);
      }
    });
    
    // Phase 4: Show training process
    timeline.add({
      duration: params.animation_duration,
      easing: 'easeInOutQuad',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Train network in batches
        const targetEpoch = Math.floor(progress * epochs);
        if (targetEpoch > currentEpoch) {
          rnn.train(sequence, target, 1);
          currentEpoch++;
        }
        
        // Draw unrolled RNN
        drawUnrolledRNN(rnn, sequence, 50, 50, 500, 250, 1);
        
        // Draw hidden states if enabled
        if (params.show_hidden_states) {
          drawHiddenStates(rnn, 50, 320, 500, 150, progress);
        }
        
        // Draw gate operations if enabled
        if (params.show_gates && params.rnn_type !== 'simple') {
          const currentStep = Math.floor(sequence.length * progress) - 1;
          if (currentStep >= 0) {
            drawGateOperations(rnn, 580, 50, 400, 250, currentStep, progress);
          }
        }
        
        // Draw loss curve if enabled
        if (params.show_training) {
          drawLossCurve(rnn.lossHistory, 580, 320, 400, 150, progress);
        }
        
        // Draw training info
        ctx.fillStyle = COLORS.text;
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Epoch: ${currentEpoch}/${epochs}`, 50, height - 80);
        ctx.fillText(`RNN Type: ${params.rnn_type.toUpperCase()}`, 50, height - 60);
        ctx.fillText(`Hidden Size: ${params.hidden_size}`, 50, height - 40);
        ctx.fillText(`Loss: ${rnn.lossHistory.length > 0 ? rnn.lossHistory[rnn.lossHistory.length - 1].toFixed(4) : 'N/A'}`, 50, height - 20);
      }
    });
    
    // Phase 5: Final reveal with all elements
    timeline.add({
      duration: 2000,
      easing: 'easeOutCubic',
      onUpdate: (progress) => {
        ctx.clearRect(0, 0, width, height);
        
        // Draw all elements
        drawUnrolledRNN(rnn, sequence, 50, 50, 400, 200, 1);
        drawRolledRNN(rnn, 480, 50, 400, 200, 1);
        drawHiddenStates(rnn, 50, 270, 400, 150, 1);
        
        if (params.rnn_type !== 'simple') {
          drawGateOperations(rnn, 480, 270, 400, 150, Math.floor(sequence.length / 2), 1);
        }
        
        drawLossCurve(rnn.lossHistory, 50, 440, 830, 150, 1);
        
        // Draw final stats
        ctx.fillStyle = COLORS.text;
        ctx.font = '16px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Final RNN Type: ${params.rnn_type.toUpperCase()}`, 50, height - 80);
        ctx.fillText(`Hidden Size: ${params.hidden_size}`, 50, height - 60);
        ctx.fillText(`Sequence Length: ${params.sequence_length}`, 50, height - 40);
        ctx.fillText(`Final Loss: ${rnn.lossHistory[rnn.lossHistory.length - 1].toFixed(4)}`, 50, height - 20);
      }
    });
    
    // Start the animation
    timeline.play();
  };
  
  // Handle different visualization types
  switch (visualizationType) {
    case 'unrolled':
      params.show_rolled = false;
      params.show_hidden_states = false;
      params.show_gates = false;
      params.show_training = false;
      animateRNN();
      break;
      
    case 'rolled':
      params.show_unrolled = false;
      params.show_hidden_states = false;
      params.show_gates = false;
      params.show_training = false;
      animateRNN();
      break;
      
    case 'hidden-states':
      params.show_unrolled = true;
      params.show_rolled = false;
      params.show_gates = false;
      params.show_training = false;
      params.show_hidden_states = true;
      animateRNN();
      break;
      
    case 'gate-operations':
      params.show_unrolled = true;
      params.show_rolled = false;
      params.show_hidden_states = false;
      params.show_training = false;
      params.show_gates = true;
      animateRNN();
      break;
      
    case 'all':
      params.show_unrolled = true;
      params.show_rolled = true;
      params.show_hidden_states = true;
      params.show_gates = true;
      params.show_training = true;
      animateRNN();
      break;
      
    default:
      animateRNN();
  }
  
  // Add enhanced controls if needed
  if (params.interactive) {
    const controls = [
      {
        label: 'RNN Type',
        type: 'select',
        options: [
          { value: 'simple', label: 'Simple RNN', selected: params.rnn_type === 'simple' },
          { value: 'lstm', label: 'LSTM', selected: params.rnn_type === 'lstm' },
          { value: 'gru', label: 'GRU', selected: params.rnn_type === 'gru' }
        ],
        onChange: (value) => {
          params.rnn_type = value;
          visualizeRNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Hidden Size',
        type: 'range',
        min: 8,
        max: 64,
        step: 8,
        value: params.hidden_size,
        onChange: (value) => {
          params.hidden_size = parseInt(value);
          visualizeRNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Sequence Length',
        type: 'range',
        min: 5,
        max: 20,
        step: 1,
        value: params.sequence_length,
        onChange: (value) => {
          params.sequence_length = parseInt(value);
          visualizeRNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Unrolled View',
        type: 'checkbox',
        checked: params.show_unrolled,
        onChange: (value) => {
          params.show_unrolled = value;
          visualizeRNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Rolled View',
        type: 'checkbox',
        checked: params.show_rolled,
        onChange: (value) => {
          params.show_rolled = value;
          visualizeRNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Hidden States',
        type: 'checkbox',
        checked: params.show_hidden_states,
        onChange: (value) => {
          params.show_hidden_states = value;
          visualizeRNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Gate Operations',
        type: 'checkbox',
        checked: params.show_gates,
        onChange: (value) => {
          params.show_gates = value;
          visualizeRNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Show Training',
        type: 'checkbox',
        checked: params.show_training,
        onChange: (value) => {
          params.show_training = value;
          visualizeRNN(containerId, visualizationType, params);
        }
      },
      {
        label: 'Visualization Type',
        type: 'select',
        options: [
          { value: 'unrolled', label: 'Unrolled View', selected: visualizationType === 'unrolled' },
          { value: 'rolled', label: 'Rolled View', selected: visualizationType === 'rolled' },
          { value: 'hidden-states', label: 'Hidden State Evolution', selected: visualizationType === 'hidden-states' },
          { value: 'gate-operations', label: 'Gate Operations', selected: visualizationType === 'gate-operations' },
          { value: 'all', label: 'Complete View', selected: visualizationType === 'all' }
        ],
        onChange: (value) => {
          visualizeRNN(containerId, value, params);
        }
      }
    ];
    
    DomUtils.createControls(containerId, controls, {
      controlsContainer: params.controlsContainer,
      title: 'RNN Parameters',
      description: 'Adjust parameters to see how they affect the recurrent neural network.'
    });
  }
}

// Multilayer Perceptron (MLP) Visualizer
function visualizeMLP(containerId, type, params) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Clear previous visualization
    container.innerHTML = '';
    
    // Create SVG canvas
    const width = container.clientWidth;
    const height = container.clientHeight;
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .style('background', '#1a1a2e');
    
    // Create animation timeline
    const timeline = gsap.timeline({repeat: 0, repeatDelay: 1});
    
    // Network architecture visualization
    if (type === 'network-architecture' || type === 'all') {
        visualizeMLPArchitecture(svg, width, height, params, timeline);
    }
    
    // Forward propagation visualization
    if (type === 'forward-propagation' || type === 'all') {
        visualizeMLPForwardPropagation(svg, width, height, params, timeline);
    }
    
    // Backward propagation visualization
    if (type === 'backward-propagation' || type === 'all') {
        visualizeMLPBackwardPropagation(svg, width, height, params, timeline);
    }
    
    // Decision boundary visualization
    if (type === 'decision-boundary' || type === 'all') {
        visualizeMLPDecisionBoundary(svg, width, height, params, timeline);
    }
    
    // Add cinematic captions
    addCinematicCaptions(container, type, timeline);
};

// Helper functions for MLP visualization
function visualizeMLPArchitecture(svg, width, height, params, timeline) {
    const { n_layers, n_neurons } = params;
    const layerSpacing = width / (n_layers + 1);
    const maxNeurons = Math.max(...n_neurons);
    
    // Create layers
    for (let i = 0; i <= n_layers; i++) {
        const x = layerSpacing * (i + 1);
        const neuronCount = i === 0 ? n_neurons[0] : (i === n_layers ? n_neurons[n_layers - 1] : n_neurons[i]);
        const neuronSpacing = height / (neuronCount + 1);
        
        // Add layer caption
        const layerName = i === 0 ? 'Input' : (i === n_layers ? 'Output' : `Hidden ${i}`);
        svg.append('text')
            .attr('x', x)
            .attr('y', 30)
            .attr('text-anchor', 'middle')
            .attr('fill', '#ffffff')
            .attr('opacity', 0)
            .text(layerName)
            .call(el => timeline.to(el.node(), {
                opacity: 1, duration: 1, delay: i * 0.5
            }, '+=0.2'));
        
        // Create neurons
        for (let j = 0; j < neuronCount; j++) {
            const y = neuronSpacing * (j + 1);
            const neuron = svg.append('circle')
                .attr('cx', x)
                .attr('cy', y)
                .attr('r', 10)
                .attr('fill', '#4cc9f0')
                .attr('stroke', '#ffffff')
                .attr('stroke-width', 2)
                .attr('opacity', 0);
            
            timeline.to(neuron.node(), {
                opacity: 1, duration: 0.8, delay: (i * 0.3) + (j * 0.1)
            }, '+=0.1');
            
            // Connect to previous layer
            if (i > 0) {
                const prevNeuronCount = i === 1 ? n_neurons[0] : n_neurons[i - 1];
                const prevNeuronSpacing = height / (prevNeuronCount + 1);
                
                for (let k = 0; k < prevNeuronCount; k++) {
                    const prevY = prevNeuronSpacing * (k + 1);
                    const connection = svg.append('line')
                        .attr('x1', x - layerSpacing)
                        .attr('y1', prevY)
                        .attr('x2', x)
                        .attr('y2', y)
                        .attr('stroke', 'rgba(255, 255, 255, 0.3)')
                        .attr('stroke-width', 1)
                        .attr('opacity', 0);
                    
                    timeline.to(connection.node(), {
                        opacity: 0.3, duration: 0.5, delay: (i * 0.2)
                    }, '+=0.1');
                }
            }
        }
    }
}

function visualizeMLPForwardPropagation(svg, width, height, params, timeline) {
    const { n_layers, n_neurons } = params;
    const layerSpacing = width / (n_layers + 1);
    
    // Animate signal flow through layers
    for (let i = 0; i < n_layers; i++) {
        const startX = layerSpacing * (i + 1);
        const endX = layerSpacing * (i + 2);
        const neuronCount = n_neurons[i];
        const nextNeuronCount = i === n_layers - 1 ? n_neurons[n_layers] : n_neurons[i + 1];
        const neuronSpacing = height / (neuronCount + 1);
        const nextNeuronSpacing = height / (nextNeuronCount + 1);
        
        for (let j = 0; j < neuronCount; j++) {
            const startY = neuronSpacing * (j + 1);
            
            for (let k = 0; k < nextNeuronCount; k++) {
                const endY = nextNeuronSpacing * (k + 1);
                
                // Create signal pulse
                const pulse = svg.append('circle')
                    .attr('cx', startX)
                    .attr('cy', startY)
                    .attr('r', 4)
                    .attr('fill', '#f72585')
                    .attr('opacity', 0);
                
                timeline.to(pulse.node(), {
                    opacity: 1, duration: 0.3
                }, `+=${(i * 0.5) + (j * 0.1)}`);
                
                timeline.to(pulse.node(), {
                    cx: endX,
                    cy: endY,
                    duration: 1.5,
                    ease: 'power2.out'
                }, `+=0.1`);
                
                timeline.to(pulse.node(), {
                    opacity: 0, duration: 0.3
                }, `+=0.1`);
                
                // Activate receiving neuron
                const neuron = svg.select(`circle[cx="${endX}"][cy="${endY}"]`);
                timeline.to(neuron.node(), {
                    fill: '#7209b7',
                    duration: 0.3
                }, `+=0.1`);
                
                timeline.to(neuron.node(), {
                    fill: '#4cc9f0',
                    duration: 0.8
                }, `+=0.2`);
            }
        }
    }
}

function visualizeMLPBackwardPropagation(svg, width, height, params, timeline) {
    const { n_layers, n_neurons } = params;
    const layerSpacing = width / (n_layers + 1);
    
    // Animate gradient flow backward through layers
    for (let i = n_layers; i > 0; i--) {
        const startX = layerSpacing * (i + 1);
        const endX = layerSpacing * i;
        const neuronCount = i === n_layers ? n_neurons[n_layers] : n_neurons[i];
        const prevNeuronCount = n_neurons[i - 1];
        const neuronSpacing = height / (neuronCount + 1);
        const prevNeuronSpacing = height / (prevNeuronCount + 1);
        
        for (let j = 0; j < neuronCount; j++) {
            const startY = neuronSpacing * (j + 1);
            
            for (let k = 0; k < prevNeuronCount; k++) {
                const endY = prevNeuronSpacing * (k + 1);
                
                // Create gradient pulse
                const pulse = svg.append('circle')
                    .attr('cx', startX)
                    .attr('cy', startY)
                    .attr('r', 4)
                    .attr('fill', '#3a0ca3')
                    .attr('opacity', 0);
                
                timeline.to(pulse.node(), {
                    opacity: 1, duration: 0.3
                }, `+=${((n_layers - i) * 0.5) + (j * 0.1)}`);
                
                timeline.to(pulse.node(), {
                    cx: endX,
                    cy: endY,
                    duration: 1.5,
                    ease: 'power2.out'
                }, `+=0.1`);
                
                timeline.to(pulse.node(), {
                    opacity: 0, duration: 0.3
                }, `+=0.1`);
                
                // Highlight connection weight update
                const connection = svg.select(`line[x1="${endX}"][y1="${endY}"][x2="${startX}"][y2="${startY}"]`);
                timeline.to(connection.node(), {
                    stroke: '#4cc9f0',
                    strokeWidth: 2,
                    duration: 0.5
                }, `+=0.1`);
                
                timeline.to(connection.node(), {
                    stroke: 'rgba(255, 255, 255, 0.3)',
                    strokeWidth: 1,
                    duration: 0.8
                }, `+=0.3`);
            }
        }
    }
}

function visualizeMLPDecisionBoundary(svg, width, height, params, timeline) {
    const { n_classes } = params;
    const padding = 50;
    const plotWidth = width - padding * 2;
    const plotHeight = height - padding * 2;
    
    // Create plot area
    const plot = svg.append('g')
        .attr('transform', `translate(${padding}, ${padding})`)
        .attr('opacity', 0);
    
    timeline.to(plot.node(), {
        opacity: 1, duration: 1
    }, '+=0.5');
    
    // Generate sample data
    const data = [];
    for (let i = 0; i < 200; i++) {
        data.push({
            x: Math.random() * plotWidth,
            y: Math.random() * plotHeight,
            class: Math.floor(Math.random() * n_classes)
        });
    }
    
    // Plot data points
    const colors = ['#f72585', '#4cc9f0', '#7209b7', '#3a0ca3', '#560bad'];
    data.forEach((point, i) => {
        const dot = plot.append('circle')
            .attr('cx', point.x)
            .attr('cy', point.y)
            .attr('r', 5)
            .attr('fill', colors[point.class])
            .attr('opacity', 0);
        
        timeline.to(dot.node(), {
            opacity: 1, duration: 0.5, delay: i * 0.01
        }, '+=0.2');
    });
    
    // Animate decision boundary evolution
    for (let epoch = 0; epoch < 5; epoch++) {
        const boundary = plot.append('path')
            .attr('d', generateDecisionBoundaryPath(epoch, plotWidth, plotHeight, n_classes))
            .attr('fill', 'none')
            .attr('stroke', '#ffffff')
            .attr('stroke-width', 2)
            .attr('opacity', 0);
        
        timeline.to(boundary.node(), {
            opacity: 0.7, duration: 1
        }, `+=${epoch * 0.5}`);
        
        if (epoch < 4) {
            timeline.to(boundary.node(), {
                opacity: 0, duration: 0.5
            }, `+=0.3`);
        }
    }
}

function generateDecisionBoundaryPath(epoch, width, height, n_classes) {
    // Simplified decision boundary path generation
    // In a real implementation, this would be based on actual MLP weights
    const complexity = 0.2 + (epoch * 0.15);
    let path = `M 0 ${height / 2}`;
    
    for (let x = 0; x <= width; x += 10) {
        const y = height / 2 + Math.sin(x * complexity * 0.1) * 100 * complexity;
        path += ` L ${x} ${y}`;
    }
    
    path += ` L ${width} ${height} L 0 ${height} Z`;
    return path;
}

// Transformer Visualizer
function visualizeTransformer(containerId, type, params) {
    const container = document.getElementById(containerId);
    if (!container) return;
    
    // Clear previous visualization
    container.innerHTML = '';
    
    // Create SVG canvas
    const width = container.clientWidth;
    const height = container.clientHeight;
    const svg = d3.select(container)
        .append('svg')
        .attr('width', width)
        .attr('height', height)
        .style('background', '#1a1a2e');
    
    // Create animation timeline
    const timeline = gsap.timeline({repeat: 0, repeatDelay: 1});
    
    // Architecture overview visualization
    if (type === 'architecture' || type === 'all') {
        visualizeTransformerArchitecture(svg, width, height, params, timeline);
    }
    
    // Attention patterns visualization
    if (type === 'attention' || type === 'all') {
        visualizeTransformerAttention(svg, width, height, params, timeline);
    }
    
    // Encoder detail visualization
    if (type === 'encoder-detail' || type === 'all') {
        visualizeTransformerEncoder(svg, width, height, params, timeline);
    }
    
    // Decoder detail visualization
    if (type === 'decoder-detail' || type === 'all') {
        visualizeTransformerDecoder(svg, width, height, params, timeline);
    }
    
    // Add cinematic captions
    addCinematicCaptions(container, type, timeline);
};

// Helper functions for Transformer visualization
function visualizeTransformerArchitecture(svg, width, height, params, timeline) {
    const { n_layers, d_model } = params;
    const centerX = width / 2;
    const centerY = height / 2;
    const encoderBlocks = [];
    const decoderBlocks = [];
    
    // Create encoder stack
    for (let i = 0; i < n_layers; i++) {
        const y = centerY - 150 + (i * 60);
        const block = svg.append('rect')
            .attr('x', centerX - 200)
            .attr('y', y)
            .attr('width', 120)
            .attr('height', 40)
            .attr('rx', 5)
            .attr('fill', '#3a0ca3')
            .attr('stroke', '#4cc9f0')
            .attr('stroke-width', 2)
            .attr('opacity', 0);
        
        encoderBlocks.push(block);
        
        timeline.to(block.node(), {
            opacity: 1, duration: 0.8, delay: i * 0.3
        }, '+=0.2');
        
        // Add encoder label
        if (i === 0) {
            svg.append('text')
                .attr('x', centerX - 140)
                .attr('y', y - 15)
                .attr('text-anchor', 'middle')
                .attr('fill', '#ffffff')
                .attr('opacity', 0)
                .text('Encoder')
                .call(el => timeline.to(el.node(), {
                    opacity: 1, duration: 0.8, delay: i * 0.3
                }, '+=0.2'));
        }
    }
    
    // Create decoder stack
    for (let i = 0; i < n_layers; i++) {
        const y = centerY - 150 + (i * 60);
        const block = svg.append('rect')
            .attr('x', centerX + 80)
            .attr('y', y)
            .attr('width', 120)
            .attr('height', 40)
            .attr('rx', 5)
            .attr('fill', '#7209b7')
            .attr('stroke', '#4cc9f0')
            .attr('stroke-width', 2)
            .attr('opacity', 0);
        
        decoderBlocks.push(block);
        
        timeline.to(block.node(), {
            opacity: 1, duration: 0.8, delay: i * 0.3
        }, '+=0.2');
        
        // Add decoder label
        if (i === 0) {
            svg.append('text')
                .attr('x', centerX + 140)
                .attr('y', y - 15)
                .attr('text-anchor', 'middle')
                .attr('fill', '#ffffff')
                .attr('opacity', 0)
                .text('Decoder')
                .call(el => timeline.to(el.node(), {
                    opacity: 1, duration: 0.8, delay: i * 0.3
                }, '+=0.2'));
        }
    }
    
    // Show information flow between encoder and decoder
    for (let i = 0; i < n_layers; i++) {
        const y = centerY - 150 + (i * 60) + 20;
        
        // Connection from encoder to decoder
        const connection = svg.append('line')
            .attr('x1', centerX - 80)
            .attr('y1', y)
            .attr('x2', centerX + 80)
            .attr('y2', y)
            .attr('stroke', '#4cc9f0')
            .attr('stroke-width', 2)
            .attr('stroke-dasharray', '5,5')
            .attr('opacity', 0);
        
        timeline.to(connection.node(), {
            opacity: 0.7, duration: 0.8, delay: i * 0.3
        }, '+=0.5');
        
        // Animate information flow along connection
        const pulse = svg.append('circle')
            .attr('cx', centerX - 80)
            .attr('cy', y)
            .attr('r', 5)
            .attr('fill', '#f72585')
            .attr('opacity', 0);
        
        timeline.to(pulse.node(), {
            opacity: 1, duration: 0.3
        }, `+=${i * 0.5 + 0.5}`);
        
        timeline.to(pulse.node(), {
            cx: centerX + 80,
            duration: 1.5,
            ease: 'power2.out'
        }, `+=0.1`);
        
        timeline.to(pulse.node(), {
            opacity: 0, duration: 0.3
        }, `+=0.1`);
    }
    
    // Add input and output embeddings
    const inputEmbedding = svg.append('rect')
        .attr('x', centerX - 350)
        .attr('y', centerY - 20)
        .attr('width', 100)
        .attr('height', 40)
        .attr('rx', 5)
        .attr('fill', '#560bad')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('opacity', 0);
    
    timeline.to(inputEmbedding.node(), {
        opacity: 1, duration: 0.8
    }, '+=0.5');
    
    svg.append('text')
        .attr('x', centerX - 300)
        .attr('y', centerY - 30)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('opacity', 0)
        .text('Input Embeddings')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.8
        }, '+=0.5'));
    
    const outputEmbedding = svg.append('rect')
        .attr('x', centerX + 250)
        .attr('y', centerY - 20)
        .attr('width', 100)
        .attr('height', 40)
        .attr('rx', 5)
        .attr('fill', '#560bad')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('opacity', 0);
    
    timeline.to(outputEmbedding.node(), {
        opacity: 1, duration: 0.8
    }, '+=0.5');
    
    svg.append('text')
        .attr('x', centerX + 300)
        .attr('y', centerY - 30)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('opacity', 0)
        .text('Output')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.8
        }, '+=0.5'));
    
    // Show input to encoder flow
    const inputFlow = svg.append('line')
        .attr('x1', centerX - 250)
        .attr('y1', centerY)
        .attr('x2', centerX - 200)
        .attr('y2', centerY - 130)
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('opacity', 0);
    
    timeline.to(inputFlow.node(), {
        opacity: 0.7, duration: 0.8
    }, '+=0.5');
    
    const inputPulse = svg.append('circle')
        .attr('cx', centerX - 250)
        .attr('cy', centerY)
        .attr('r', 5)
        .attr('fill', '#f72585')
        .attr('opacity', 0);
    
    timeline.to(inputPulse.node(), {
        opacity: 1, duration: 0.3
    }, '+=0.6');
    
    timeline.to(inputPulse.node(), {
        cx: centerX - 200,
        cy: centerY - 130,
        duration: 1.5,
        ease: 'power2.out'
    }, '+=0.1');
    
    timeline.to(inputPulse.node(), {
        opacity: 0, duration: 0.3
    }, '+=0.1');
    
    // Show output from decoder flow
    const outputFlow = svg.append('line')
        .attr('x1', centerX + 200)
        .attr('y1', centerY - 130)
        .attr('x2', centerX + 250)
        .attr('y2', centerY)
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('opacity', 0);
    
    timeline.to(outputFlow.node(), {
        opacity: 0.7, duration: 0.8
    }, '+=0.5');
    
    const outputPulse = svg.append('circle')
        .attr('cx', centerX + 200)
        .attr('cy', centerY - 130)
        .attr('r', 5)
        .attr('fill', '#f72585')
        .attr('opacity', 0);
    
    timeline.to(outputPulse.node(), {
        opacity: 1, duration: 0.3
    }, '+=1.5');
    
    timeline.to(outputPulse.node(), {
        cx: centerX + 250,
        cy: centerY,
        duration: 1.5,
        ease: 'power2.out'
    }, '+=0.1');
    
    timeline.to(outputPulse.node(), {
        opacity: 0, duration: 0.3
    }, '+=0.1');
}

function visualizeTransformerAttention(svg, width, height, params, timeline) {
    const { n_heads, sequence_length } = params;
    const centerX = width / 2;
    const centerY = height / 2;
    const matrixSize = Math.min(width, height) * 0.6;
    const cellSize = matrixSize / sequence_length;
    
    // Create attention matrix
    const matrixGroup = svg.append('g')
        .attr('transform', `translate(${centerX - matrixSize/2}, ${centerY - matrixSize/2})`)
        .attr('opacity', 0);
    
    timeline.to(matrixGroup.node(), {
        opacity: 1, duration: 1
    }, '+=0.5');
    
    // Create matrix cells
    for (let i = 0; i < sequence_length; i++) {
        for (let j = 0; j < sequence_length; j++) {
            const attentionValue = Math.random(); // Simulated attention weight
            
            const cell = matrixGroup.append('rect')
                .attr('x', i * cellSize)
                .attr('y', j * cellSize)
                .attr('width', cellSize)
                .attr('height', cellSize)
                .attr('fill', d3.interpolatePlasma(attentionValue))
                .attr('opacity', 0.8)
                .attr('stroke', '#ffffff')
                .attr('stroke-width', 0.5)
                .attr('opacity', 0);
            
            timeline.to(cell.node(), {
                opacity: 0.8, duration: 0.3, delay: (i + j) * 0.05
            }, '+=0.2');
        }
    }
    
    // Add row and column labels (tokens)
    for (let i = 0; i < sequence_length; i++) {
        // Row labels (query tokens)
        matrixGroup.append('text')
            .attr('x', -10)
            .attr('y', i * cellSize + cellSize/2)
            .attr('text-anchor', 'end')
            .attr('fill', '#ffffff')
            .attr('dominant-baseline', 'middle')
            .attr('font-size', '10px')
            .attr('opacity', 0)
            .text(`Q${i+1}`)
            .call(el => timeline.to(el.node(), {
                opacity: 1, duration: 0.5, delay: i * 0.1
            }, '+=0.3'));
        
        // Column labels (key tokens)
        matrixGroup.append('text')
            .attr('x', i * cellSize + cellSize/2)
            .attr('y', -10)
            .attr('text-anchor', 'middle')
            .attr('fill', '#ffffff')
            .attr('dominant-baseline', 'bottom')
            .attr('font-size', '10px')
            .attr('opacity', 0)
            .text(`K${i+1}`)
            .call(el => timeline.to(el.node(), {
                opacity: 1, duration: 0.5, delay: i * 0.1
            }, '+=0.3'));
    }
    
    // Add title
    matrixGroup.append('text')
        .attr('x', matrixSize / 2)
        .attr('y', -30)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('font-size', '14px')
        .attr('opacity', 0)
        .text('Attention Weights Matrix')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.8
        }, '+=0.3'));
    
    // Animate attention patterns
    for (let head = 0; head < n_heads; head++) {
        // Highlight specific attention patterns for each head
        for (let i = 0; i < sequence_length; i++) {
            const focusCell = Math.floor(Math.random() * sequence_length);
            
            // Highlight row
            for (let j = 0; j < sequence_length; j++) {
                const cell = matrixGroup.select(`rect[x="${i * cellSize}"][y="${j * cellSize}"]`);
                timeline.to(cell.node(), {
                    fill: '#f72585',
                    duration: 0.3
                }, `+=${(head * sequence_length * 0.5) + (i * 0.3)}`);
                
                timeline.to(cell.node(), {
                    fill: d3.interpolatePlasma(Math.random()),
                    duration: 0.5
                }, `+=0.2`);
            }
            
            // Highlight specific cell with stronger effect
            const focusedCell = matrixGroup.select(`rect[x="${i * cellSize}"][y="${focusCell * cellSize}"]`);
            timeline.to(focusedCell.node(), {
                stroke: '#4cc9f0',
                strokeWidth: 3,
                duration: 0.3
            }, `+=${(head * sequence_length * 0.5) + (i * 0.3) + 0.1}`);
            
            timeline.to(focusedCell.node(), {
                stroke: '#ffffff',
                strokeWidth: 0.5,
                duration: 0.5
            }, `+=0.2`);
        }
        
        // Add head caption
        if (n_heads > 1) {
            svg.append('text')
                .attr('x', centerX)
                .attr('y', centerY + matrixSize/2 + 40)
                .attr('text-anchor', 'middle')
                .attr('fill', '#4cc9f0')
                .attr('font-size', '12px')
                .attr('opacity', 0)
                .text(`Head ${head + 1}/${n_heads}`)
                .call(el => timeline.to(el.node(), {
                    opacity: 1, duration: 0.5
                }, `+=${head * sequence_length * 0.5}`));
            
            timeline.to(el => {
                if (el && el.node) {
                    el.node().opacity = 0;
                }
            }, {
                duration: 0.5,
                delay: 1
            }, `+=${head * sequence_length * 0.5 + 1}`);
        }
    }
}

function visualizeTransformerEncoder(svg, width, height, params, timeline) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Create encoder block
    const encoderBlock = svg.append('g')
        .attr('transform', `translate(${centerX - 100}, ${centerY - 100})`)
        .attr('opacity', 0);
    
    timeline.to(encoderBlock.node(), {
        opacity: 1, duration: 1
    }, '+=0.5');
    
    // Self-attention layer
    const selfAttention = encoderBlock.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 120)
        .attr('height', 40)
        .attr('rx', 5)
        .attr('fill', '#3a0ca3')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('opacity', 0);
    
    timeline.to(selfAttention.node(), {
        opacity: 1, duration: 0.8
    }, '+=0.2');
    
    encoderBlock.append('text')
        .attr('x', 60)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('dominant-baseline', 'middle')
        .attr('opacity', 0)
        .text('Self-Attention')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.8
        }, '+=0.2'));
    
    // Feed-forward layer
    const feedForward = encoderBlock.append('rect')
        .attr('x', 0)
        .attr('y', 80)
        .attr('width', 120)
        .attr('height', 40)
        .attr('rx', 5)
        .attr('fill', '#3a0ca3')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('opacity', 0);
    
    timeline.to(feedForward.node(), {
        opacity: 1, duration: 0.8, delay: 0.5
    }, '+=0.2');
    
    encoderBlock.append('text')
        .attr('x', 60)
        .attr('y', 100)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('dominant-baseline', 'middle')
        .attr('opacity', 0)
        .text('Feed Forward')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.8, delay: 0.5
        }, '+=0.2'));
    
    // Residual connections
    const residual1 = encoderBlock.append('path')
        .attr('d', 'M 120 20 C 140 20, 140 0, 160 0 L 180 0 M 180 0 L 180 100 C 180 120, 160 120, 160 120 L 120 120')
        .attr('fill', 'none')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .attr('opacity', 0);
    
    timeline.to(residual1.node(), {
        opacity: 0.7, duration: 0.8, delay: 1
    }, '+=0.2');
    
    const residual2 = encoderBlock.append('path')
        .attr('d', 'M 120 100 C 140 100, 140 120, 160 120 L 180 120')
        .attr('fill', 'none')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .attr('opacity', 0);
    
    timeline.to(residual2.node(), {
        opacity: 0.7, duration: 0.8, delay: 1.5
    }, '+=0.2');
    
    // Add layer normalization indicators
    const addNorm1 = encoderBlock.append('circle')
        .attr('cx', 130)
        .attr('cy', 60)
        .attr('r', 8)
        .attr('fill', '#7209b7')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 1)
        .attr('opacity', 0);
    
    timeline.to(addNorm1.node(), {
        opacity: 1, duration: 0.5, delay: 2
    }, '+=0.2');
    
    encoderBlock.append('text')
        .attr('x', 130)
        .attr('y', 60)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '10px')
        .attr('opacity', 0)
        .text('LN')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.5, delay: 2
        }, '+=0.2'));
    
    const addNorm2 = encoderBlock.append('circle')
        .attr('cx', 130)
        .attr('cy', 140)
        .attr('r', 8)
        .attr('fill', '#7209b7')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 1)
        .attr('opacity', 0);
    
    timeline.to(addNorm2.node(), {
        opacity: 1, duration: 0.5, delay: 2.5
    }, '+=0.2');
    
    encoderBlock.append('text')
        .attr('x', 130)
        .attr('y', 140)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '10px')
        .attr('opacity', 0)
        .text('LN')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.5, delay: 2.5
        }, '+=0.2'));
    
    // Animate information flow
    const inputPulse = encoderBlock.append('circle')
        .attr('cx', -20)
        .attr('cy', 20)
        .attr('r', 5)
        .attr('fill', '#f72585')
        .attr('opacity', 0);
    
    timeline.to(inputPulse.node(), {
        opacity: 1, duration: 0.3, delay: 3
    }, '+=0.2');
    
    // Flow through self-attention
    timeline.to(inputPulse.node(), {
        cx: 60,
        cy: 20,
        duration: 1,
        ease: 'power2.out'
    }, '+=0.1');
    
    // Flow to add & norm
    timeline.to(inputPulse.node(), {
        cx: 130,
        cy: 60,
        duration: 0.8,
        ease: 'power2.out'
    }, '+=0.2');
    
    // Flow to feed-forward
    timeline.to(inputPulse.node(), {
        cx: 60,
        cy: 100,
        duration: 1,
        ease: 'power2.out'
    }, '+=0.2');
    
    // Flow to second add & norm
    timeline.to(inputPulse.node(), {
        cx: 130,
        cy: 140,
        duration: 0.8,
        ease: 'power2.out'
    }, '+=0.2');
    
    // Flow out of encoder
    timeline.to(inputPulse.node(), {
        cx: 200,
        cy: 100,
        duration: 1,
        ease: 'power2.out'
    }, '+=0.2');
    
    timeline.to(inputPulse.node(), {
        opacity: 0, duration: 0.3
    }, '+=0.1');
}

function visualizeTransformerDecoder(svg, width, height, params, timeline) {
    const centerX = width / 2;
    const centerY = height / 2;
    
    // Create decoder block
    const decoderBlock = svg.append('g')
        .attr('transform', `translate(${centerX - 100}, ${centerY - 150})`)
        .attr('opacity', 0);
    
    timeline.to(decoderBlock.node(), {
        opacity: 1, duration: 1
    }, '+=0.5');
    
    // Masked self-attention layer
    const maskedAttention = decoderBlock.append('rect')
        .attr('x', 0)
        .attr('y', 0)
        .attr('width', 120)
        .attr('height', 40)
        .attr('rx', 5)
        .attr('fill', '#560bad')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('opacity', 0);
    
    timeline.to(maskedAttention.node(), {
        opacity: 1, duration: 0.8
    }, '+=0.2');
    
    decoderBlock.append('text')
        .attr('x', 60)
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('dominant-baseline', 'middle')
        .attr('opacity', 0)
        .text('Masked Attention')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.8
        }, '+=0.2'));
    
    // Encoder-decoder attention layer
    const encoderDecoderAttention = decoderBlock.append('rect')
        .attr('x', 0)
        .attr('y', 80)
        .attr('width', 120)
        .attr('height', 40)
        .attr('rx', 5)
        .attr('fill', '#560bad')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('opacity', 0);
    
    timeline.to(encoderDecoderAttention.node(), {
        opacity: 1, duration: 0.8, delay: 0.5
    }, '+=0.2');
    
    decoderBlock.append('text')
        .attr('x', 60)
        .attr('y', 100)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('dominant-baseline', 'middle')
        .attr('opacity', 0)
        .text('Encoder-Decoder Attention')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.8, delay: 0.5
        }, '+=0.2'));
    
    // Feed-forward layer
    const feedForward = decoderBlock.append('rect')
        .attr('x', 0)
        .attr('y', 160)
        .attr('width', 120)
        .attr('height', 40)
        .attr('rx', 5)
        .attr('fill', '#560bad')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('opacity', 0);
    
    timeline.to(feedForward.node(), {
        opacity: 1, duration: 0.8, delay: 1
    }, '+=0.2');
    
    decoderBlock.append('text')
        .attr('x', 60)
        .attr('y', 180)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('dominant-baseline', 'middle')
        .attr('opacity', 0)
        .text('Feed Forward')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.8, delay: 1
        }, '+=0.2'));
    
    // Residual connections
    const residual1 = decoderBlock.append('path')
        .attr('d', 'M 120 20 C 140 20, 140 0, 160 0 L 180 0 M 180 0 L 180 100 C 180 120, 160 120, 160 120 L 120 120')
        .attr('fill', 'none')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .attr('opacity', 0);
    
    timeline.to(residual1.node(), {
        opacity: 0.7, duration: 0.8, delay: 1.5
    }, '+=0.2');
    
    const residual2 = decoderBlock.append('path')
        .attr('d', 'M 120 100 C 140 100, 140 120, 160 120 L 180 120 M 180 120 L 180 180 C 180 200, 160 200, 160 200 L 120 200')
        .attr('fill', 'none')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .attr('opacity', 0);
    
    timeline.to(residual2.node(), {
        opacity: 0.7, duration: 0.8, delay: 2
    }, '+=0.2');
    
    const residual3 = decoderBlock.append('path')
        .attr('d', 'M 120 180 C 140 180, 140 200, 160 200 L 180 200')
        .attr('fill', 'none')
        .attr('stroke', '#4cc9f0')
        .attr('stroke-width', 2)
        .attr('stroke-dasharray', '5,5')
        .attr('opacity', 0);
    
    timeline.to(residual3.node(), {
        opacity: 0.7, duration: 0.8, delay: 2.5
    }, '+=0.2');
    
    // Add layer normalization indicators
    const addNorm1 = decoderBlock.append('circle')
        .attr('cx', 130)
        .attr('cy', 60)
        .attr('r', 8)
        .attr('fill', '#7209b7')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 1)
        .attr('opacity', 0);
    
    timeline.to(addNorm1.node(), {
        opacity: 1, duration: 0.5, delay: 3
    }, '+=0.2');
    
    decoderBlock.append('text')
        .attr('x', 130)
        .attr('y', 60)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '10px')
        .attr('opacity', 0)
        .text('LN')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.5, delay: 3
        }, '+=0.2'));
    
    const addNorm2 = decoderBlock.append('circle')
        .attr('cx', 130)
        .attr('cy', 140)
        .attr('r', 8)
        .attr('fill', '#7209b7')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 1)
        .attr('opacity', 0);
    
    timeline.to(addNorm2.node(), {
        opacity: 1, duration: 0.5, delay: 3.5
    }, '+=0.2');
    
    decoderBlock.append('text')
        .attr('x', 130)
        .attr('y', 140)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '10px')
        .attr('opacity', 0)
        .text('LN')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.5, delay: 3.5
        }, '+=0.2'));
    
    const addNorm3 = decoderBlock.append('circle')
        .attr('cx', 130)
        .attr('cy', 220)
        .attr('r', 8)
        .attr('fill', '#7209b7')
        .attr('stroke', '#ffffff')
        .attr('stroke-width', 1)
        .attr('opacity', 0);
    
    timeline.to(addNorm3.node(), {
        opacity: 1, duration: 0.5, delay: 4
    }, '+=0.2');
    
    decoderBlock.append('text')
        .attr('x', 130)
        .attr('y', 220)
        .attr('text-anchor', 'middle')
        .attr('fill', '#ffffff')
        .attr('dominant-baseline', 'middle')
        .attr('font-size', '10px')
        .attr('opacity', 0)
        .text('LN')
        .call(el => timeline.to(el.node(), {
            opacity: 1, duration: 0.5, delay: 4
        }, '+=0.2'));
    
    // Show encoder input to decoder
    const encoderInput = svg.append('circle')
        .attr('cx', centerX - 150)
        .attr('cy', centerY - 50)
        .attr('r', 5)
        .attr('fill', '#4cc9f0')
        .attr('opacity', 0);
    
    timeline.to(encoderInput.node(), {
        opacity: 1, duration: 0.3, delay: 4.5
    }, '+=0.2');
    
    timeline.to(encoderInput.node(), {
        cx: centerX - 50,
        cy: centerY - 100,
        duration: 1.5,
        ease: 'power2.out'
    }, '+=0.1');
    
    // Animate information flow through decoder
    const inputPulse = decoderBlock.append('circle')
        .attr('cx', -20)
        .attr('cy', 20)
        .attr('r', 5)
        .attr('fill', '#f72585')
        .attr('opacity', 0);
    
    timeline.to(inputPulse.node(), {
        opacity: 1, duration: 0.3, delay: 5
    }, '+=0.2');
    
    // Flow through masked attention
    timeline.to(inputPulse.node(), {
        cx: 60,
        cy: 20,
        duration: 1,
        ease: 'power2.out'
    }, '+=0.1');
    
    // Flow to first add & norm
    timeline.to(inputPulse.node(), {
        cx: 130,
        cy: 60,
        duration: 0.8,
        ease: 'power2.out'
    }, '+=0.2');
    
    // Flow to encoder-decoder attention (combine with encoder input)
    timeline.to(inputPulse.node(), {
        cx: 60,
        cy: 100,
        duration: 1,
        ease: 'power2.out'
    }, '+=0.2');
    
    // Merge with encoder input
    timeline.to(encoderInput.node(), {
        cx: 60,
        cy: 100,
        duration: 0.8,
        ease: 'power2.out'
    }, '+=0.1');
    
    // Flow to second add & norm
    timeline.to(inputPulse.node(), {
        cx: 130,
        cy: 140,
        duration: 0.8,
        ease: 'power2.out'
    }, '+=0.2');
    
    timeline.to(encoderInput.node(), {
        opacity: 0, duration: 0.3
    }, '+=0.1');
    
    // Flow to feed-forward
    timeline.to(inputPulse.node(), {
        cx: 60,
        cy: 180,
        duration: 1,
        ease: 'power2.out'
    }, '+=0.2');
    
    // Flow to third add & norm
    timeline.to(inputPulse.node(), {
        cx: 130,
        cy: 220,
        duration: 0.8,
        ease: 'power2.out'
    }, '+=0.2');
    
    // Flow out of decoder
    timeline.to(inputPulse.node(), {
        cx: 200,
        cy: 180,
        duration: 1,
        ease: 'power2.out'
    }, '+=0.2');
    
    timeline.to(inputPulse.node(), {
        opacity: 0, duration: 0.3
    }, '+=0.1');
    
    // Show output token generation
    const outputToken = svg.append('text')
        .attr('x', centerX + 150)
        .attr('y', centerY - 50)
        .attr('text-anchor', 'middle')
        .attr('fill', '#4cc9f0')
        .attr('font-size', '16px')
        .attr('font-weight', 'bold')
        .attr('opacity', 0)
        .text('Output');
    
    timeline.to(outputToken.node(), {
        opacity: 1, duration: 0.8, delay: 7
    }, '+=0.2');
    
    // Pulse output token
    timeline.to(outputToken.node(), {
        fill: '#f72585',
        duration: 0.5
    }, '+=0.3');
    
    timeline.to(outputToken.node(), {
        fill: '#4cc9f0',
        duration: 0.5
    }, '+=0.3');
}

// Helper function to add cinematic captions
function addCinematicCaptions(container, type, timeline) {
    const captions = {
        'mlp': {
            'network-architecture': [
                { text: 'Multilayer Perceptron Architecture', delay: 0, duration: 3 },
                { text: 'Input → Hidden → Output Layers', delay: 3, duration: 3 },
                { text: 'Each neuron connected to all neurons in next layer', delay: 6, duration: 4 }
            ],
            'forward-propagation': [
                { text: 'Forward Propagation', delay: 0, duration: 3 },
                { text: 'Signals flow from input to output', delay: 3, duration: 3 },
                { text: 'Activation functions transform the signals', delay: 6, duration: 4 }
            ],
            'backward-propagation': [
                { text: 'Backpropagation', delay: 0, duration: 3 },
                { text: 'Gradients flow backward through the network', delay: 3, duration: 3 },
                { text: 'Weights are updated to minimize loss', delay: 6, duration: 4 }
            ],
            'decision-boundary': [
                { text: 'Decision Boundary Evolution', delay: 0, duration: 3 },
                { text: 'Network learns to separate classes', delay: 3, duration: 3 },
                { text: 'Boundary becomes more complex with training', delay: 6, duration: 4 }
            ],
            'all': [
                { text: 'Multilayer Perceptron in Action', delay: 0, duration: 4 },
                { text: 'Architecture → Forward Pass → Backward Pass → Decision Boundary', delay: 4, duration: 5 },
                { text: 'Neural networks learn hierarchical representations', delay: 9, duration: 5 }
            ]
        },
        'transformer': {
            'architecture': [
                { text: 'Transformer Architecture', delay: 0, duration: 3 },
                { text: 'Encoder-Decoder Structure with Self-Attention', delay: 3, duration: 4 },
                { text: 'Parallel processing of sequences', delay: 7, duration: 3 }
            ],
            'attention': [
                { text: 'Attention Mechanism', delay: 0, duration: 3 },
                { text: 'Weights represent token relationships', delay: 3, duration: 3 },
                { text: 'Different heads capture different patterns', delay: 6, duration: 4 }
            ],
            'encoder-detail': [
                { text: 'Encoder Block', delay: 0, duration: 3 },
                { text: 'Self-Attention + Feed Forward with Residual Connections', delay: 3, duration: 4 },
                { text: 'Layer normalization stabilizes training', delay: 7, duration: 3 }
            ],
            'decoder-detail': [
                { text: 'Decoder Block', delay: 0, duration: 3 },
                { text: 'Masked Self-Attention + Encoder-Decoder Attention', delay: 3, duration: 4 },
                { text: 'Generates output tokens autoregressively', delay: 7, duration: 3 }
            ],
            'all': [
                { text: 'Transformer Complete View', delay: 0, duration: 4 },
                { text: 'Architecture → Attention → Encoder → Decoder → Output', delay: 4, duration: 5 },
                { text: 'Revolutionized NLP with parallel sequence processing', delay: 9, duration: 5 }
            ]
        }
    };
    
    const algorithm = container.id.includes('mlp') ? 'mlp' : 'transformer';
    const captionData = captions[algorithm][type];
    
    if (captionData) {
        captionData.forEach((caption, index) => {
            const captionEl = document.createElement('div');
            captionEl.className = 'cinematic-caption';
            captionEl.style.cssText = `
                position: absolute;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                color: white;
                font-size: 24px;
                font-weight: bold;
                text-align: center;
                text-shadow: 0 0 10px rgba(0, 0, 0, 0.7);
                opacity: 0;
                pointer-events: none;
                z-index: 1000;
                width: 80%;
            `;
            captionEl.textContent = caption.text;
            container.appendChild(captionEl);
            
            timeline.to(captionEl, {
                opacity: 1, duration: 1, delay: caption.delay
            });
            
            timeline.to(captionEl, {
                opacity: 0, duration: 1, delay: caption.duration
            });
        });
    }
}

// Helper function for attention flow animation
function animateAttentionFlow(svg, encoderX, decoderX, height, params) {
    // Create flowing particles between encoder and decoder
    for (let i = 0; i < 5; i++) {
        setTimeout(() => {
            const yPos = 150 + Math.random() * (height - 250);
            
            const particle = svg.append('circle')
                .attr('cx', encoderX + 50)
                .attr('cy', yPos)
                .attr('r', 3)
                .style('fill', '#f72585')
                .style('opacity', 0);
                
            particle.transition()
                .duration(2000)
                .attr('cx', decoderX - 50)
                .style('opacity', 0.8)
                .on('end', () => particle.remove());
                
        }, i * 400);
    }
}

// Helper functions (would be in a separate utilities file)
function generateClassificationData(n_samples, n_features, noise) {
    const data = [];
    for (let i = 0; i < n_samples; i++) {
        const x = Math.random();
        const y = Math.random();
        // Simple classification pattern
        const label = (x - 0.5) * (y - 0.5) > 0 ? 1 : 0;
        // Add some noise
        const noisyLabel = Math.random() < noise ? 1 - label : label;
        
        data.push({ x, y, label: noisyLabel });
    }
    return data;
}

function generateRegressionData(n_samples) {
    const points = [];
    const trueFunction = x => 0.3 + 0.4 * Math.sin(x * 2 * Math.PI);
    
    for (let i = 0; i < n_samples; i++) {
        const x = Math.random();
        const y = trueFunction(x) + (Math.random() - 0.5) * 0.2;
        points.push({ x, y });
    }
    
    return { points, trueFunction };
}


// =============================================
// Global Visualizers Object
// =============================================

window.visualizers = {
  "linear-regression": visualizeLinearRegression,
  "logistic-regression": visualizeLogisticRegression,
  "decision-tree": visualizeDecisionTree,
  "qda": visualizeQDA,
  "lda": visualizeLDA,
  "naive-bayes": visualizeNaiveBayes,
  "random-forest": visualizeRandomForest,
  "knn": visualizeKNN,
  "svm": visualizeSVM,
  "k-means": visualizeKMeans,
  "hierarchical-clustering": visualizeHierarchicalClustering,
  "gaussian-mixture-models": visualizeGaussianMixtureModels,
  "pca": visualizePCA,
  "dbscan": visualizeDBSCAN,
  "bagging": visualizeBagging,
  "adaboost": visualizeAdaBoost,
  "gbm": visualizeGBM,
  "xgboost": visualizeXGBoost,
  "neural-network": visualizeNeuralNetwork,
  "cnn": visualizeCNN,
  "rnn": visualizeRNN,
  "mlp": visualizeMLP,
  "transformer": visualizeTransformer,
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
    visualizeQDA,
    visualizeLDA,
    visualizeNaiveBayes,
    visualizeRandomForest,
    visualizeKNN,
    visualizeSVM,
    visualizeKMeans,
    visualizeHierarchicalClustering,
    visualizeGaussianMixtureModels,
    visualizePCA,
    visualizeDBSCAN,
    visualizeBagging,
    visualizeAdaBoost,
    visualizeGBM,
    visualizeXGBoost,
    visualizeNeuralNetwork,
    visualizeCNN,
    visualizeRNN,
    visualizeMLP,
    visualizeTransformer,
  };
}
