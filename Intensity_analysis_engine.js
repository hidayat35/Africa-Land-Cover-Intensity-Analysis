// ======================================================================
// === 1. SETUP & CONFIGURATION =========================================
// ======================================================================

// *** IMPORTANT: Ensure 'roi' is imported. If named 'table', uncomment below: ***
// var roi = table; 

// --- 1.1 Region Setup ---
var roiInfo = roi.limit(1).getInfo();
var props = roiInfo.features.length > 0 ? roiInfo.features[0].properties : {};
var propNames = Object.keys(props);

var regionCol = 'LAB'; // Default
if (propNames.indexOf('LAB') === -1) {
  var candidates = propNames.filter(function(name) { 
    return ['label', 'name', 'acronym', 'region', 'id'].indexOf(name.toLowerCase()) > -1; 
  });
  if (candidates.length > 0) regionCol = candidates[0];
}

var targetRegions = ['EAF', 'MED', 'SAF', 'SAH', 'WAF'];
var allOptions = ['Whole Africa'].concat(targetRegions);

// --- 1.2 Land Cover Setup ---
var lc_names = ['Cropland', 'Forest', 'Shrubland', 'Grassland', 'Tundra', 'Wetland', 'Impervious', 'Bare', 'Water', 'Snow/Ice'];
var lc_indices = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']; 

// Reclassification Logic
var from_classes = [10, 11, 12, 20, 51, 52, 61, 62, 71, 72, 81, 82, 91, 92, 120, 121, 122, 130, 140, 181, 182, 183, 184, 185, 186, 187, 190, 150, 152, 153, 200, 201, 202, 210, 220, 0];
var to_classes =   [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 6, 6, 6, 6, 7, 8, 8, 8, 8, 8, 8, 9, 10, 0];

function reclassify(image) {
  var remapped = image.remap(from_classes, to_classes).rename('landcover');
  return remapped.updateMask(remapped.neq(0));
}

// ======================================================================
// === 2. DATA PREPARATION ==============================================
// ======================================================================

var five_year_col = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/five-years-map');
var five_year_img = five_year_col.mosaic();

var getFiveYear = function(band, year) {
  return reclassify(five_year_img.select(band)).set('year', year).set('system:time_start', ee.Date.fromYMD(year, 1, 1).millis());
};

var img1985 = getFiveYear('b1', 1985);
var img1990 = getFiveYear('b2', 1990);
var img1995 = getFiveYear('b3', 1995);
var pre2000_col = ee.ImageCollection.fromImages([img1985, img1990, img1995]);

var annual_mosaic = ee.ImageCollection('projects/sat-io/open-datasets/GLC-FCS30D/annual').mosaic();
var years_annual = ee.List.sequence(2000, 2022);
var annual_col = ee.ImageCollection.fromImages(years_annual.map(function(year) {
  year = ee.Number(year);
  var band_index = year.subtract(1999);
  var band_name = ee.String('b').cat(band_index.format('%d'));
  return reclassify(annual_mosaic.select([band_name])).set('year', year).set('system:time_start', ee.Date.fromYMD(year, 1, 1).millis());
}));

var full_col = pre2000_col.merge(annual_col).sort('system:time_start');
var yearsToAnalyze = [1985, 1990, 1995, 2000, 2005, 2010, 2015, 2020, 2022];

// ======================================================================
// === 3. UI PANEL ======================================================
// ======================================================================

var panel = ui.Panel({style: {width: '500px', padding: '10px'}});
ui.root.add(panel);

panel.add(ui.Label('Intensity Analysis: Africa', {fontWeight: 'bold', fontSize: '20px'}));

var regionSelect = ui.Select({items: allOptions, value: 'Whole Africa', placeholder: 'Select Region'});
panel.add(ui.Label('1. Select Region:', {fontWeight: 'bold'}));
panel.add(regionSelect);

var scaleSlider = ui.Slider({min: 250, max: 10000, value: 5000, step: 250, style: {width: '300px'}});
panel.add(ui.Label('2. Scale (m). Lower = Slower but Accurate.', {fontWeight: 'bold'}));
panel.add(scaleSlider);

var runButton = ui.Button('Calculate', runAnalysis);
panel.add(runButton);

var resultsPanel = ui.Panel();
panel.add(resultsPanel);

// ======================================================================
// === 4. ANALYSIS LOGIC ================================================
// ======================================================================

function runAnalysis() {
  resultsPanel.clear();
  resultsPanel.add(ui.Label('Initializing...', {color: 'gray'}));
  
  var selectedRegion = regionSelect.getValue();
  var scale = scaleSlider.getValue();
  
  // 1. Resolve Geometry
  var geom;
  if (selectedRegion === 'Whole Africa') {
    geom = roi.filter(ee.Filter.inList(regionCol, targetRegions)).geometry();
  } else {
    geom = roi.filter(ee.Filter.eq(regionCol, selectedRegion)).geometry();
  }
  
  Map.centerObject(geom);
  Map.layers().reset();
  Map.addLayer(geom, {color: 'red'}, 'ROI Outline', false);
  
  // 2. Data Check
  var check = img1985.reduceRegion({
    reducer: ee.Reducer.count(),
    geometry: geom,
    scale: scale,
    maxPixels: 1e13
  });
  
  check.evaluate(function(stats) {
    if (!stats || stats.landcover === 0) {
      resultsPanel.clear();
      resultsPanel.add(ui.Label('ERROR: No valid data found in ' + selectedRegion + ' at Scale ' + scale + 'm.', {color: 'red', fontWeight: 'bold'}));
      resultsPanel.add(ui.Label('Try reducing the scale (e.g. 1000m) or checking ROI overlap.'));
      return;
    } 
    
    resultsPanel.add(ui.Label('Data Check Passed: ' + stats.landcover + ' pixels found.', {fontSize: '10px', color: 'green'}));
    performIntensityAnalysis(geom, scale);
  });
}

function performIntensityAnalysis(geom, scale) {
  var analysisYears = ee.List(yearsToAnalyze);
  var numIntervals = analysisYears.length().subtract(1);
  var intervalIndices = ee.List.sequence(0, numIntervals.subtract(1));

  var results = intervalIndices.map(function(i) {
    var y1 = ee.Number(analysisYears.get(i));
    var y2 = ee.Number(analysisYears.get(ee.Number(i).add(1)));
    var duration = y2.subtract(y1);
    
    var img1 = full_col.filter(ee.Filter.eq('year', y1)).first().clip(geom);
    var img2 = full_col.filter(ee.Filter.eq('year', y2)).first().clip(geom);
    
    var transImg = img1.multiply(100).add(img2).rename('trans');
    
    var histogram = transImg.reduceRegion({
      reducer: ee.Reducer.frequencyHistogram(),
      geometry: geom,
      scale: scale,
      maxPixels: 1e13,
      tileScale: 4
    }).get('trans');
    
    var dict = ee.Dictionary(histogram);
    var keys = dict.keys();
    
    var parsed = keys.map(function(k) {
      k = ee.String(k);
      var val = ee.Number(dict.get(k));
      var val_num = ee.Number.parse(k);
      var from = val_num.divide(100).floor().toInt();
      var to = val_num.mod(100).toInt();
      return ee.Feature(null, {'from': from, 'to': to, 'pixels': val, 'isPersist': from.eq(to)});
    });
    
    var fc = ee.FeatureCollection(parsed);
    var totalPx = fc.aggregate_sum('pixels');
    var persistPx = fc.filter(ee.Filter.eq('isPersist', 1)).aggregate_sum('pixels');
    var changePx = ee.Algorithms.If(totalPx.gt(0), ee.Number(totalPx).subtract(persistPx), 0);
    
    var intensity = ee.Algorithms.If(totalPx.gt(0), ee.Number(changePx).divide(totalPx).divide(duration).multiply(100), 0);
    
    var classStats = ee.List.sequence(1, 10).map(function(c) {
      c = ee.Number(c);
      var startPx = fc.filter(ee.Filter.eq('from', c)).aggregate_sum('pixels');
      var endPx = fc.filter(ee.Filter.eq('to', c)).aggregate_sum('pixels');
      var pPx = fc.filter(ee.Filter.and(ee.Filter.eq('from', c), ee.Filter.eq('to', c))).aggregate_sum('pixels');
      
      var loss = ee.Number(startPx).subtract(pPx);
      var gain = ee.Number(endPx).subtract(pPx);
      
      var lossI = ee.Algorithms.If(startPx.gt(0), loss.divide(duration).divide(startPx).multiply(100), 0);
      var gainI = ee.Algorithms.If(endPx.gt(0), gain.divide(duration).divide(endPx).multiply(100), 0);
      
      return ee.Dictionary({
        'cid': c, 'gI': gainI, 'lI': lossI
      });
    });

    return ee.Feature(null, {
      'label': ee.String(y1.format('%d')).cat('-').cat(y2.format('%d')),
      'int': intensity,
      'stats': classStats,
      'dur': duration,
      'chgPx': changePx,
      'totPx': totalPx
    });
  });

  var resCol = ee.FeatureCollection(results);
  
  // --- VISUALIZATION ---
  resCol.evaluate(function(fc) {
    if (!fc || fc.features.length === 0) {
      resultsPanel.add(ui.Label('Error: Empty results.', {color: 'red'}));
      return;
    }
    
    var feats = fc.features;
    var intervalLabels = [];
    var yInt = [];
    var yU = [];
    
    var totalChg = 0; var totalDur = 0; var totalTot = 0;
    
    // Process data for Interval Chart
    feats.forEach(function(f) {
      var p = f.properties;
      intervalLabels.push(p.label);
      yInt.push(p.int);
      totalChg += p.chgPx;
      totalDur += p.dur;
      totalTot += p.totPx;
    });
    
    var meanTotal = totalTot / feats.length;
    var uGlobal = (meanTotal > 0) ? (totalChg / meanTotal) / totalDur * 100 : 0;
    
    for (var k=0; k<yInt.length; k++) yU.push(uGlobal);
    
    // --- Chart 1: Interval ---
    var chart1 = ui.Chart.array.values(
      [yInt, yU], 
      1, 
      intervalLabels
    ).setChartType('ColumnChart')
      .setSeriesNames(['Annual Intensity', 'Uniform'])
      .setOptions({
        title: 'Interval Level Intensity',
        vAxis: {title: '% Change / Year'},
        hAxis: {title: 'Interval'},
        series: {
          0: {color: 'black'},
          1: {type: 'line', color: 'red', lineDashStyle: [4, 4], pointSize: 0, lineWidth: 2}
        }
      });
      
    resultsPanel.add(ui.Label('Interval Level:', {fontWeight: 'bold', fontSize: '16px', margin: '20px 0 0 0'}));
    resultsPanel.add(chart1);
    resultsPanel.add(ui.Label('Global Uniform Intensity (U): ' + uGlobal.toFixed(2) + '%', {color: 'red'}));
    
    // --- SECTION: Category Level (Interactive) ---
    resultsPanel.add(ui.Label('Category Level:', {fontWeight: 'bold', fontSize: '16px', margin: '20px 0 0 0'}));
    resultsPanel.add(ui.Label('Select interval to view Category Intensity:'));

    var catChartPanel = ui.Panel(); // Container for the chart
    
    // Dropdown for selecting "Average" or specific interval
    var selectItems = ['Average (All Years)'].concat(intervalLabels);
    
    var intervalSelect = ui.Select({
      items: selectItems,
      value: 'Average (All Years)',
      onChange: function(selection) {
        updateCategoryChart(selection);
      }
    });
    
    resultsPanel.add(intervalSelect);
    resultsPanel.add(catChartPanel);
    
    // Function to render/update the Category Chart based on selection
    var updateCategoryChart = function(selection) {
      catChartPanel.clear();
      
      var cLab = []; var gVal = []; var lVal = []; var uRef = [];
      var referenceLineValue = 0;
      var titleText = "";
      
      if (selection === 'Average (All Years)') {
        // --- Calculate Average Stats ---
        referenceLineValue = uGlobal; // Reference is Global Uniform
        titleText = 'Category Level (Average)';
        
        var aggStats = {}; 
        feats.forEach(function(f) {
          f.properties.stats.forEach(function(s) {
            if (!aggStats[s.cid]) aggStats[s.cid] = {g:[], l:[]};
            aggStats[s.cid].g.push(s.gI);
            aggStats[s.cid].l.push(s.lI);
          });
        });
        
        lc_indices.forEach(function(idxStr, i) {
          var cid = parseInt(idxStr);
          if (aggStats[cid]) {
            cLab.push(lc_names[i]);
            var gSum = aggStats[cid].g.reduce(function(a, b) { return a + b; }, 0);
            var lSum = aggStats[cid].l.reduce(function(a, b) { return a + b; }, 0);
            
            // Avoid division by zero
            var count = aggStats[cid].g.length;
            gVal.push(count > 0 ? gSum / count : 0);
            lVal.push(count > 0 ? lSum / count : 0);
            uRef.push(referenceLineValue);
          }
        });
        
      } else {
        // --- Calculate Specific Interval Stats ---
        // 1. Find the feature for this interval
        var feat = feats.filter(function(f) { return f.properties.label === selection; })[0];
        
        if (feat) {
          // Reference is that Interval's Intensity (St)
          referenceLineValue = feat.properties.int;
          titleText = 'Category Level (' + selection + ')';
          
          // Create a map of stats for quick lookup
          var statsMap = {};
          feat.properties.stats.forEach(function(s) {
            statsMap[s.cid] = s;
          });
          
          lc_indices.forEach(function(idxStr, i) {
            var cid = parseInt(idxStr);
            if (statsMap[cid]) {
              cLab.push(lc_names[i]);
              gVal.push(statsMap[cid].gI);
              lVal.push(statsMap[cid].lI);
              uRef.push(referenceLineValue);
            }
          });
        }
      }

      // Render Chart
      var chartHeight = Math.max(400, cLab.length * 30);
      
      var chart = ui.Chart.array.values(
        [gVal, lVal, uRef],
        1,
        cLab
      ).setChartType('BarChart')
        .setSeriesNames(['Gain', 'Loss', 'Ref Intensity'])
        .setOptions({
          title: titleText,
          height: chartHeight + 'px',
          chartArea: {height: '85%', width: '70%'},
          hAxis: {title: 'Annual Intensity (%)'},
          vAxis: {title: 'Category', textStyle: {fontSize: 12, bold: true}},
          series: {
            0: {color: 'green'},
            1: {color: 'red'},
            2: {type: 'line', color: 'black', lineDashStyle: [4, 4], pointSize: 0, lineWidth: 2}
          }
        });
        
      catChartPanel.add(chart);
      
      var legendLabel = (selection === 'Average (All Years)') ? 
        'Black dashed line = Uniform Intensity (U)' : 
        'Black dashed line = Interval Intensity (' + selection + ')';
        
      catChartPanel.add(ui.Label(legendLabel, {fontSize: '11px', color: 'gray'}));
    };
    
    // Initial Render
    updateCategoryChart('Average (All Years)');
  });
}
