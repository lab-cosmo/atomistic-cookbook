(function() {
    // Calculate days since Unix epoch (January 1, 1970)
    var millisecondsPerDay = 1000 * 60 * 60 * 24;
    var today = new Date();
    var utcMidnight = Date.UTC(today.getFullYear(), today.getMonth(), today.getDate());
    var epoch = Date.UTC(1970, 0, 1);
    var daysSinceEpoch = Math.floor((utcMidnight - epoch) / millisecondsPerDay);

    // Calculate index based on the day of the year
    var index = daysSinceEpoch % examplesData.length;

    // Get the example
    var example = examplesData[index];

    // Build the HTML code
    var html = '';
    html += '<div class="grid">';
    html += '  <div class="grid-item">';
    html += '    <div class="card" style="text-align: center; box-shadow: md;">';
    html += '      <a href="' + example.ref + '" class="doc">';
    html += '        <h3 class="card-title">' + example.title + '</h3>';
    html += '        <img src="' + example.thumbnail + '" alt="' + example.description + '" class="gallery-img">';
    html += '      </a>';
    html += '      <p>' + example.description + "</p>";
    html += '    </div>';
    html += '  </div>';
    html += '</div>';

    // Insert the HTML into the page
    var container = document.getElementById('daily-thumbnail');
    if (container) {
        container.innerHTML = html;
    }
})();
