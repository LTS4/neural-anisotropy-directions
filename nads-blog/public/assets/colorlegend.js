function renderColorBar (svg, color, x, y, width, height) {
  const colorBar = svg.append('g')
    .attr('transform', `translate(${x}, ${y})`)
  const colorScale = d3.scaleLinear()
    .domain([height, 0])
    .range([0, 1])
  for (let y = 0; y < height; ++y) {
    colorBar.append('rect')
      .attr('x', 0)
      .attr('y', y)
      .attr('width', width)
      .attr('height', 1)
      .attr('fill', color(colorScale(y)))
      .attr('stroke', 'none')
  }

  return colorBar
}